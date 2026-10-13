const { GoogleGenAI } = require('@google/genai');
const logger = require('../core/logger').createServiceLogger('LLM');
const config = require('../core/config');
const { promptLoader } = require('../../prompt-loader');

class LLMService {
  constructor() {
    this.client = null;
    this.model = null;
    this.provider = 'gemini';
    this.isInitialized = false;
    this.requestCount = 0;
    this.errorCount = 0;
    
    this.initializeClient();
  }

  initializeClient() {
    const requested = String(
      process.env.LLM_PROVIDER || config.get('llm.provider') || 'gemini'
    ).trim().toLowerCase();
    this.provider = requested === 'openrouter' ? 'openrouter' : 'gemini';
    this.client = null;
    this.isInitialized = false;

    if (this.provider === 'openrouter') {
      this._initializeOpenRouterClient();
      return;
    }

    this._initializeGeminiClient();
  }

  _initializeGeminiClient() {
    const apiKey = config.getApiKey('GEMINI');
    
    if (!apiKey || apiKey === 'your-api-key-here' || apiKey === 'your_gemini_api_key_here') {
      logger.warn('Gemini API key not configured', { 
        keyExists: !!apiKey,
        isPlaceholder: apiKey === 'your-api-key-here' || apiKey === 'your_gemini_api_key_here'
      });
      return;
    }

    try {
      this.client = new GoogleGenAI({ apiKey });
      
      // Use the configured model name (default: gemini-3.5-flash)
      this.model = config.get('llm.gemini.model');
      this.isInitialized = true;
      
      logger.info('Gemini AI client initialized successfully', {
        model: this.model
      });
    } catch (error) {
      logger.error('Failed to initialize Gemini client', { 
        error: error.message 
      });
    }
  }

  _initializeOpenRouterClient() {
    const apiKey = config.getApiKey('OPENROUTER');

    if (!apiKey || /your.*openrouter.*key/i.test(apiKey)) {
      logger.warn('OpenRouter API key not configured', {
        keyExists: !!apiKey
      });
      return;
    }

    try {
      this.model = process.env.OPENROUTER_MODEL || config.get('llm.openrouter.model');
      this.isInitialized = true;
      logger.info('OpenRouter client initialized successfully', {
        model: this.model
      });
    } catch (error) {
      logger.error('Failed to initialize OpenRouter client', {
        error: error.message
      });
    }
  }

  _notInitializedMessage() {
    return this.provider === 'openrouter'
      ? 'LLM service not initialized. Check OpenRouter API key configuration.'
      : 'LLM service not initialized. Check Gemini API key configuration.';
  }

  getGenerationConfig(overrides = {}) {
    const isOpenRouter = this.provider === 'openrouter';
    const defaults = (isOpenRouter
      ? config.get('llm.openrouter.generation')
      : config.get('llm.gemini.generation')) || {};
    const fallback = isOpenRouter
      ? {
          temperature: 0.7,
          topP: 0.9,
          maxOutputTokens: 4096
        }
      : {
          temperature: 0.7,
          topK: 40,
          topP: 0.95,
          maxOutputTokens: 4096,
          thinkingConfig: { thinkingBudget: 0 }
        };

    const merged = { ...fallback, ...defaults, ...overrides };
    return Object.fromEntries(
      Object.entries(merged).filter(([, value]) => value !== undefined && value !== null)
    );
  }

  applyGenerationDefaults(request, overrides = {}) {
    request.generationConfig = this.getGenerationConfig({ ...(request.generationConfig || {}), ...overrides });
    return request;
  }

  extractTextFromCandidates(response) {
    // New @google/genai SDK exposes response.text as a convenience getter.
    if (response && typeof response.text === 'string' && response.text.trim().length > 0) {
      return {
        text: response.text.trim(),
        candidate: response.candidates?.[0] || null,
        finishReason: response.candidates?.[0]?.finishReason || null
      };
    }

    const candidates = Array.isArray(response?.candidates)
      ? response.candidates
      : Array.isArray(response)
        ? response
        : [];

    if (!candidates.length) {
      throw new Error('No candidates in Gemini response');
    }

    const candidateWithText = candidates.find(candidate => {
      const parts = candidate?.content?.parts;
      return Array.isArray(parts) && parts.some(part => typeof part.text === 'string' && part.text.trim().length > 0);
    });

    if (!candidateWithText) {
      const finishReasons = candidates.map(c => c.finishReason || 'unknown').join(', ');
      throw new Error(`No text parts in candidates. Finish reasons: ${finishReasons}`);
    }

    const textParts = candidateWithText.content.parts
      .filter(part => typeof part.text === 'string' && part.text.trim().length > 0)
      .map(part => part.text.trim());

    if (!textParts.length) {
      throw new Error(`Candidate parts missing text after filtering: ${JSON.stringify(candidateWithText)}`);
    }

    const text = textParts.join('\n');

    return {
      text,
      candidate: candidateWithText,
      finishReason: candidateWithText.finishReason || null
    };
  }

  /**
   * Process an image directly with Gemini using the active skill prompt.
   * The image buffer is sent as inlineData alongside a concise instruction.
   * For image-based queries, we include the skill prompt (e.g., DSA) as systemInstruction.
   * @param {Buffer} imageBuffer - PNG/JPEG image bytes
   * @param {string} mimeType - e.g., 'image/png' or 'image/jpeg'
   * @param {string} activeSkill - current skill (e.g. 'dsa')
   * @param {Array} sessionMemory - optional (not required for image)
   * @param {string|null} programmingLanguage - optional language context for skills that need it
   * @returns {Promise<{response: string, metadata: object}>}
   */
  async processImageWithSkill(imageBuffer, mimeType, activeSkill, sessionMemory = [], programmingLanguage = null) {
    if (!this.isInitialized) {
      throw new Error(this._notInitializedMessage());
    }

    if (!imageBuffer || !Buffer.isBuffer(imageBuffer)) {
      throw new Error('Invalid image buffer provided to processImageWithSkill');
    }

    const startTime = Date.now();
    this.requestCount++;

    try {
      // Build system instruction using the skill prompt (with optional language injection)
      const { promptLoader } = require('../../prompt-loader');
      const skillPrompt = promptLoader.getSkillPrompt(activeSkill, programmingLanguage) || '';

      // Build request with text + image parts
      const base64 = imageBuffer.toString('base64');

      const request = {
        contents: [
          {
            role: 'user',
            parts: [
              { text: this.formatImageInstruction(activeSkill, programmingLanguage) },
              { inlineData: { data: base64, mimeType } }
            ]
          }
        ]
      };

      this.applyGenerationDefaults(request);

      if (skillPrompt && skillPrompt.trim().length > 0) {
        request.systemInstruction = { parts: [{ text: skillPrompt }] };
      }

      // Execute with retries/timeout - try alternative method first for network reliability
      let responseText;
      const preferAlternative = !!config.get('llm.gemini.enableFallbackMethod');
      try {
        if (preferAlternative) {
          logger.debug('Attempting alternative HTTPS method first for reliability');
          responseText = await this.executeAlternativeRequest(request);
        } else {
          responseText = await this.executeRequest(request);
        }
      } catch (error) {
        const secondaryLabel = preferAlternative ? 'primary SDK method' : 'alternative HTTPS method';
        logger.warn(`${preferAlternative ? 'Alternative' : 'Primary'} method failed, trying ${secondaryLabel}`, { error: error.message });
        const secondaryFn = preferAlternative ? this.executeRequest.bind(this) : this.executeAlternativeRequest.bind(this);

        try {
          responseText = await secondaryFn(request);
        } catch (secondaryError) {
          logger.error('Both Gemini request methods failed', {
            firstError: error.message,
            secondError: secondaryError.message
          });
          throw secondaryError;
        }
      }

      // Enforce language in code fences if provided
      const finalResponse = programmingLanguage
        ? this.enforceProgrammingLanguage(responseText, programmingLanguage)
        : responseText;

      logger.logPerformance('LLM image processing', startTime, {
        activeSkill,
        imageSize: imageBuffer.length,
        responseLength: finalResponse.length,
        programmingLanguage: programmingLanguage || 'not specified',
        requestId: this.requestCount
      });

      return {
        response: finalResponse,
        metadata: {
          skill: activeSkill,
          programmingLanguage,
          processingTime: Date.now() - startTime,
          requestId: this.requestCount,
          usedFallback: false,
          isImageAnalysis: true,
          mimeType
        }
      };
    } catch (error) {
      this.errorCount++;
      logger.error('LLM image processing failed', {
        error: error.message,
        activeSkill,
        requestId: this.requestCount
      });

      if (config.get('llm.gemini.fallbackEnabled')) {
        return this.generateFallbackResponse('[image]', activeSkill);
      }
      throw error;
    }
  }

  async processImageWithSkillStream(imageBuffer, mimeType, activeSkill, sessionMemory = [], programmingLanguage = null, onDelta = null) {
    if (!this.isInitialized) {
      throw new Error(this._notInitializedMessage());
    }

    if (!imageBuffer || !Buffer.isBuffer(imageBuffer)) {
      throw new Error('Invalid image buffer provided to processImageWithSkillStream');
    }

    const startTime = Date.now();
    this.requestCount++;

    try {
      const { promptLoader } = require('../../prompt-loader');
      const skillPrompt = promptLoader.getSkillPrompt(activeSkill, programmingLanguage) || '';
      const base64 = imageBuffer.toString('base64');

      const geminiRequest = {
        contents: [
          {
            role: 'user',
            parts: [
              { text: this.formatImageInstruction(activeSkill, programmingLanguage) },
              { inlineData: { data: base64, mimeType } }
            ]
          }
        ]
      };
      this.applyGenerationDefaults(geminiRequest);
      if (skillPrompt && skillPrompt.trim().length > 0) {
        geminiRequest.systemInstruction = { parts: [{ text: skillPrompt }] };
      }

      const fullText = await this.executeStreamingRequest(geminiRequest, (delta) => {
        if (typeof onDelta === 'function' && delta) {
          onDelta(delta);
        }
      });

      const finalResponse = programmingLanguage
        ? this.enforceProgrammingLanguage(fullText, programmingLanguage)
        : fullText;

      logger.logPerformance('LLM image streaming', startTime, {
        activeSkill,
        imageSize: imageBuffer.length,
        responseLength: finalResponse.length,
        requestId: this.requestCount
      });

      return {
        response: finalResponse,
        metadata: {
          skill: activeSkill,
          programmingLanguage,
          processingTime: Date.now() - startTime,
          requestId: this.requestCount,
          usedFallback: false,
          streamed: true,
          isImageAnalysis: true,
          mimeType
        }
      };
    } catch (error) {
      logger.warn('Streaming image analysis failed, falling back to non-streaming', {
        error: error.message,
        requestId: this.requestCount
      });
      return this.processImageWithSkill(imageBuffer, mimeType, activeSkill, sessionMemory, programmingLanguage);
    }
  }

  formatImageInstruction(activeSkill, programmingLanguage) {
    const langNote = programmingLanguage ? ` If it is a coding question, use only ${programmingLanguage.toUpperCase()} for any code.` : '';
    return `This is a screenshot from a live interview. Identify what is being asked or shown, then give the best possible answer the candidate can read and use immediately. If it is a coding problem, extract it and provide a complete, runnable solution.${langNote}`;
  }

  async processTextWithSkill(text, activeSkill, sessionMemory = [], programmingLanguage = null) {
    if (!this.isInitialized) {
      throw new Error(this._notInitializedMessage());
    }

    const startTime = Date.now();
    this.requestCount++;
    
    try {
      logger.info('Processing text with LLM', {
        activeSkill,
        textLength: text.length,
        hasSessionMemory: sessionMemory.length > 0,
        programmingLanguage: programmingLanguage || 'not specified',
        requestId: this.requestCount
      });

      const geminiRequest = this.buildGeminiRequest(text, activeSkill, sessionMemory, programmingLanguage);

      const preferAlternative = !!config.get('llm.gemini.enableFallbackMethod');
      let response;
      try {
        if (preferAlternative) {
          logger.debug('Attempting alternative HTTPS method first for text processing');
          response = await this.executeAlternativeRequest(geminiRequest);
        } else {
          response = await this.executeRequest(geminiRequest);
        }
      } catch (error) {
        const secondaryLabel = preferAlternative ? 'primary SDK method' : 'alternative HTTPS method';
        logger.warn(`${preferAlternative ? 'Alternative' : 'Primary'} method failed, trying ${secondaryLabel}`, {
          error: error.message,
          requestId: this.requestCount
        });
        const secondaryFn = preferAlternative ? this.executeRequest.bind(this) : this.executeAlternativeRequest.bind(this);
        try {
          response = await secondaryFn(geminiRequest);
        } catch (secondaryError) {
          logger.error('Both Gemini request methods failed for text processing', {
            firstError: error.message,
            secondError: secondaryError.message,
            requestId: this.requestCount
          });
          throw secondaryError;
        }
      }
      
      // Enforce language in code fences if programmingLanguage specified
      const finalResponse = programmingLanguage
        ? this.enforceProgrammingLanguage(response, programmingLanguage)
        : response;

      logger.logPerformance('LLM text processing', startTime, {
        activeSkill,
        textLength: text.length,
        responseLength: finalResponse.length,
        programmingLanguage: programmingLanguage || 'not specified',
        requestId: this.requestCount
      });

      return {
        response: finalResponse,
        metadata: {
          skill: activeSkill,
          programmingLanguage,
          processingTime: Date.now() - startTime,
          requestId: this.requestCount,
          usedFallback: false
        }
      };
    } catch (error) {
      this.errorCount++;
      logger.error('LLM processing failed', {
        error: error.message,
        activeSkill,
        programmingLanguage: programmingLanguage || 'not specified',
        requestId: this.requestCount
      });

      if (config.get('llm.gemini.fallbackEnabled')) {
        return this.generateFallbackResponse(text, activeSkill);
      }

      throw error;
    }
  }

  async processTextWithSkillStream(text, activeSkill, sessionMemory = [], programmingLanguage = null, onDelta = null) {
    if (!this.isInitialized) {
      throw new Error(this._notInitializedMessage());
    }

    const startTime = Date.now();
    this.requestCount++;

    try {
      const geminiRequest = this.buildGeminiRequest(text, activeSkill, sessionMemory, programmingLanguage);

      const fullText = await this.executeStreamingRequest(geminiRequest, (delta) => {
        if (typeof onDelta === 'function' && delta) {
          onDelta(delta);
        }
      });

      const finalResponse = programmingLanguage
        ? this.enforceProgrammingLanguage(fullText, programmingLanguage)
        : fullText;

      logger.logPerformance('LLM text streaming', startTime, {
        activeSkill,
        textLength: text.length,
        responseLength: finalResponse.length,
        requestId: this.requestCount
      });

      return {
        response: finalResponse,
        metadata: {
          skill: activeSkill,
          programmingLanguage,
          processingTime: Date.now() - startTime,
          requestId: this.requestCount,
          usedFallback: false,
          streamed: true
        }
      };
    } catch (error) {
      logger.warn('Streaming text failed, falling back to non-streaming', {
        error: error.message,
        requestId: this.requestCount
      });
      return this.processTextWithSkill(text, activeSkill, sessionMemory, programmingLanguage);
    }
  }

  async processTranscriptionWithIntelligentResponse(text, activeSkill, sessionMemory = [], programmingLanguage = null) {
    if (!this.isInitialized) {
      throw new Error(this._notInitializedMessage());
    }

    const startTime = Date.now();
    this.requestCount++;
    
    try {
      logger.info('Processing transcription with intelligent response', {
        activeSkill,
        textLength: text.length,
        hasSessionMemory: sessionMemory.length > 0,
        programmingLanguage: programmingLanguage || 'not specified',
        requestId: this.requestCount
      });

      const geminiRequest = this.buildIntelligentTranscriptionRequest(text, activeSkill, sessionMemory, programmingLanguage);

      const preferAlternative = !!config.get('llm.gemini.enableFallbackMethod');
      let response;
      try {
        if (preferAlternative) {
          logger.debug('Attempting alternative HTTPS method first for transcription processing');
          response = await this.executeAlternativeRequest(geminiRequest);
        } else {
          response = await this.executeRequest(geminiRequest);
        }
      } catch (error) {
        const secondaryLabel = preferAlternative ? 'primary SDK method' : 'alternative HTTPS method';
        logger.warn(`${preferAlternative ? 'Alternative' : 'Primary'} method failed, trying ${secondaryLabel}`, {
          error: error.message,
          requestId: this.requestCount
        });
        const secondaryFn = preferAlternative ? this.executeRequest.bind(this) : this.executeAlternativeRequest.bind(this);
        try {
          response = await secondaryFn(geminiRequest);
        } catch (secondaryError) {
          logger.error('Both Gemini request methods failed for transcription processing', {
            firstError: error.message,
            secondError: secondaryError.message,
            requestId: this.requestCount
          });
          throw secondaryError;
        }
      }
      
      // Enforce language in code fences if programmingLanguage specified
      const finalResponse = programmingLanguage
        ? this.enforceProgrammingLanguage(response, programmingLanguage)
        : response;

      logger.logPerformance('LLM transcription processing', startTime, {
        activeSkill,
        textLength: text.length,
        responseLength: finalResponse.length,
        programmingLanguage: programmingLanguage || 'not specified',
        requestId: this.requestCount
      });

      return {
        response: finalResponse,
        metadata: {
          skill: activeSkill,
          programmingLanguage,
          processingTime: Date.now() - startTime,
          requestId: this.requestCount,
          usedFallback: false,
          isTranscriptionResponse: true
        }
      };
    } catch (error) {
      this.errorCount++;
      logger.error('LLM transcription processing failed', {
        error: error.message,
        activeSkill,
        programmingLanguage: programmingLanguage || 'not specified',
        requestId: this.requestCount
      });

      if (config.get('llm.gemini.fallbackEnabled')) {
        return this.generateIntelligentFallbackResponse(text, activeSkill);
      }
      
      throw error;
    }
  }

  /**
   * Normalize all triple-backtick code fences to the selected programming language tag.
   * Does not alter the inner code; only ensures fence language tags are correct.
   */
  enforceProgrammingLanguage(text, programmingLanguage) {
    try {
      if (!text || !programmingLanguage) return text;
      const norm = String(programmingLanguage).toLowerCase();
      const fenceTagMap = { cpp: 'cpp', c: 'c', python: 'python', java: 'java', javascript: 'javascript', js: 'javascript' };
      const fenceTag = fenceTagMap[norm] || norm || 'text';

      // Replace all triple-backtick fences' language token with the selected tag
      const replacedBackticks = text.replace(/```([^\n]*)\n/g, (match, info) => {
        const current = (info || '').trim();
        // If already the desired fenceTag as the first token, keep as is
        if (current.split(/\s+/)[0].toLowerCase() === fenceTag) return match;
        return '```' + fenceTag + '\n';
      });

      // Optionally normalize tildes fences to backticks with correct tag
      const normalizedTildes = replacedBackticks.replace(/~~~([^\n]*)\n/g, () => '```' + fenceTag + '\n');

      return normalizedTildes;
    } catch (_) {
      return text;
    }
  }

  buildGeminiRequest(text, activeSkill, sessionMemory, programmingLanguage) {
    // Check if we have the new conversation history format
    const sessionManager = require('../managers/session.manager');
    
    if (sessionManager && typeof sessionManager.getConversationHistory === 'function') {
      const conversationHistory = sessionManager.getConversationHistory(15);
      const skillContext = sessionManager.getSkillContext(activeSkill, programmingLanguage);
      return this.buildGeminiRequestWithHistory(text, activeSkill, conversationHistory, skillContext, programmingLanguage);
    }

    // Fallback to old method for compatibility - now with programming language support
    const requestComponents = promptLoader.getRequestComponents(
      activeSkill, 
      text, 
      sessionMemory,
      programmingLanguage
    );

    const request = {
      contents: []
    };

    this.applyGenerationDefaults(request);

    // Use the skill prompt that already has programming language injected
    if (requestComponents.shouldUseModelMemory && requestComponents.skillPrompt) {
      request.systemInstruction = {
        parts: [{ text: requestComponents.skillPrompt }]
      };
      
      logger.debug('Using language-enhanced system instruction for skill', {
        skill: activeSkill,
        programmingLanguage: programmingLanguage || 'not specified',
        promptLength: requestComponents.skillPrompt.length,
        requiresProgrammingLanguage: requestComponents.requiresProgrammingLanguage
      });
    }

    request.contents.push({
      role: 'user',
      parts: [{ text: this.formatUserMessage(text, activeSkill) }]
    });

    return request;
  }

  buildGeminiRequestWithHistory(text, activeSkill, conversationHistory, skillContext, programmingLanguage) {
    const request = {
      contents: []
    };

    this.applyGenerationDefaults(request);

    // Use the skill prompt from context (which may already include programming language)
    if (skillContext.skillPrompt) {
      request.systemInstruction = {
        parts: [{ text: skillContext.skillPrompt }]
      };
      
      logger.debug('Using skill context prompt as system instruction', {
        skill: activeSkill,
        programmingLanguage: programmingLanguage || 'not specified',
        promptLength: skillContext.skillPrompt.length,
        requiresProgrammingLanguage: skillContext.requiresProgrammingLanguage || false,
        hasLanguageInjection: programmingLanguage && skillContext.requiresProgrammingLanguage
      });
    }

    // Add conversation history (excluding system messages) with validation
    const conversationContents = conversationHistory
      .filter(event => {
        return event.role !== 'system' && 
               event.content && 
               typeof event.content === 'string' && 
               event.content.trim().length > 0;
      })
      .map(event => {
        const content = event.content.trim();
        return {
          role: event.role === 'model' ? 'model' : 'user',
          parts: [{ text: content }]
        };
      });

    // Add the conversation history
    request.contents.push(...conversationContents);

    // Format and validate the current user input
    const formattedMessage = this.formatUserMessage(text, activeSkill);
    if (!formattedMessage || formattedMessage.trim().length === 0) {
      throw new Error('Failed to format user message or message is empty');
    }

    // Add the current user input
    request.contents.push({
      role: 'user',
      parts: [{ text: formattedMessage }]
    });

    logger.debug('Built Gemini request with conversation history', {
      skill: activeSkill,
      programmingLanguage: programmingLanguage || 'not specified',
      historyLength: conversationHistory.length,
      totalContents: request.contents.length,
      hasSystemInstruction: !!request.systemInstruction,
      requiresProgrammingLanguage: skillContext.requiresProgrammingLanguage || false
    });

    return request;
  }

  buildIntelligentTranscriptionRequest(text, activeSkill, sessionMemory, programmingLanguage) {
    // Validate input text first
    const cleanText = text && typeof text === 'string' ? text.trim() : '';
    if (!cleanText) {
      throw new Error('Empty or invalid transcription text provided to buildIntelligentTranscriptionRequest');
    }

    // Check if we have the new conversation history format
    const sessionManager = require('../managers/session.manager');
    
    if (sessionManager && typeof sessionManager.getConversationHistory === 'function') {
      const conversationHistory = sessionManager.getConversationHistory(10);
      const skillContext = sessionManager.getSkillContext(activeSkill, programmingLanguage);
      return this.buildIntelligentTranscriptionRequestWithHistory(cleanText, activeSkill, conversationHistory, skillContext, programmingLanguage);
    }

    // Fallback to basic intelligent request
    const request = {
      contents: []
    };

    this.applyGenerationDefaults(request);

    // Add intelligent filtering system instruction
    const intelligentPrompt = this.getIntelligentTranscriptionPrompt(activeSkill, programmingLanguage);
    if (!intelligentPrompt) {
      throw new Error('Failed to generate intelligent transcription prompt');
    }

    request.systemInstruction = {
      parts: [{ text: intelligentPrompt }]
    };

    request.contents.push({
      role: 'user',
      parts: [{ text: cleanText }]
    });

    logger.debug('Built basic intelligent transcription request', {
      skill: activeSkill,
      programmingLanguage: programmingLanguage || 'not specified',
      textLength: cleanText.length,
      hasSystemInstruction: !!request.systemInstruction
    });

    return request;
  }

  buildIntelligentTranscriptionRequestWithHistory(text, activeSkill, conversationHistory, skillContext, programmingLanguage) {
    const request = {
      contents: []
    };

    this.applyGenerationDefaults(request);

  // For chat/transcription messages, DO NOT include the full skill prompt; use only the intelligent filter prompt
  const intelligentPrompt = this.getIntelligentTranscriptionPrompt(activeSkill, programmingLanguage);
  request.systemInstruction = { parts: [{ text: intelligentPrompt }] };

    // Add recent conversation history (excluding system messages) with validation
    const conversationContents = conversationHistory
      .filter(event => {
        // Filter out system messages and ensure content exists and is valid
        return event.role !== 'system' && 
               event.content && 
               typeof event.content === 'string' && 
               event.content.trim().length > 0;
      })
      .slice(-8) // Keep last 8 exchanges for context
      .map(event => {
        const content = event.content.trim();
        if (!content) {
          logger.warn('Empty content found in conversation history', { event });
          return null;
        }
        return {
          role: event.role === 'model' ? 'model' : 'user',
          parts: [{ text: content }]
        };
      })
      .filter(content => content !== null); // Remove any null entries

    // Add the conversation history
    request.contents.push(...conversationContents);

    // Validate and add the current transcription
    const cleanText = text && typeof text === 'string' ? text.trim() : '';
    if (!cleanText) {
      throw new Error('Empty or invalid transcription text provided');
    }

    request.contents.push({
      role: 'user',
      parts: [{ text: cleanText }]
    });

    // Ensure we have at least one content item
    if (request.contents.length === 0) {
      throw new Error('No valid content to send to Gemini API');
    }

    logger.debug('Built intelligent transcription request with conversation history', {
      skill: activeSkill,
      programmingLanguage: programmingLanguage || 'not specified',
      historyLength: conversationHistory.length,
      totalContents: request.contents.length,
      hasSkillPrompt: !!skillContext.skillPrompt,
      cleanTextLength: cleanText.length,
      requiresProgrammingLanguage: skillContext.requiresProgrammingLanguage || false
    });

    return request;
  }

  getIntelligentTranscriptionPrompt(activeSkill, programmingLanguage) {
    let prompt = `# Real-Time Interview Copilot

You are an invisible interview copilot listening live to a conversation. The text below is what was just said out loud (usually by the interviewer). Work out what the speaker wants and answer so the candidate can read your reply on screen and say it naturally.

## Classify the input first
- Greeting, small talk, filler, or a statement that needs no answer -> reply with one very short, natural acknowledgement and nothing else.
- Behavioral / situational question ("Tell me about a time…") -> answer in the first person, Situation → Task → Action → Result.
- Coding / DSA / algorithms question -> state the idea and complexity in a line or two, then give a complete, runnable implementation.
- System design, product, data, sales, or general knowledge question -> give a crisp, structured answer.
- Follow-up -> stay consistent with the conversation history.

## Answer rules
- Lead with the answer; never restate the question or pad with filler.
- Be concise: a few tight bullets, or one small code block. The candidate is reading this while talking.
- Be correct and confident. Cover the key points and any important trade-off.
- Never reveal that you are an AI or a tool, and never narrate that you are helping.
- If the input is unrelated to the interview, keep the reply to a single short line so you don't waste attention.`;

    // Add programming language context if provided.
    if (programmingLanguage) {
      const lang = String(programmingLanguage).toLowerCase();
      const languageMap = { cpp: 'C++', c: 'C', python: 'Python', java: 'Java', javascript: 'JavaScript', js: 'JavaScript' };
      const fenceTagMap = { cpp: 'cpp', c: 'c', python: 'python', java: 'java', javascript: 'javascript', js: 'javascript' };
      const languageTitle = languageMap[lang] || (lang.charAt(0).toUpperCase() + lang.slice(1));
      const fenceTag = fenceTagMap[lang] || lang || 'text';
      prompt += `\n\nCODING QUESTIONS: When (and only when) the question is a coding problem, respond in ${languageTitle} using triple backticks tagged exactly \`\`\`${fenceTag}\`\`\`, with the optimal solution and its time and space complexity. For every other kind of question, answer in plain prose.`;
    }

    prompt += `

If the input is a coding or DSA problem statement and contains no code, produce a complete, runnable solution in the selected programming language without asking for more details, and end with the final implementation in a properly tagged code block.`;

    return prompt;
  }

  formatUserMessage(text, activeSkill) {
    return `Interview question:\n${text}`;
  }

  async executeRequest(geminiRequest) {
    if (this.provider === 'openrouter') {
      return this._executeOpenRouterRequest(geminiRequest);
    }

    const maxRetries = config.get('llm.gemini.maxRetries');
    const timeout = config.get('llm.gemini.timeout');
    const primaryModel = this.model;
    const fallbackModels = config.get('llm.gemini.fallbackModels') || [];
    const modelsToTry = [primaryModel, ...fallbackModels];

    logger.debug('Executing Gemini request', {
      hasModel: !!this.model,
      hasClient: !!this.client,
      requestKeys: Object.keys(geminiRequest),
      timeout,
      maxRetries,
      modelsToTry,
      nodeVersion: process.version,
      platform: process.platform
    });

    let lastError = null;

    for (const modelName of modelsToTry) {
      for (let attempt = 1; attempt <= maxRetries; attempt++) {
        try {
          const timeoutPromise = new Promise((_, reject) =>
            setTimeout(() => reject(new Error('Request timeout')), timeout)
          );

          logger.debug(`Gemini API attempt ${attempt} starting with model ${modelName}`, {
            timestamp: new Date().toISOString(),
            timeout,
            model: modelName
          });

          const requestPromise = this.client.models.generateContent({
            model: modelName,
            contents: geminiRequest.contents,
            config: geminiRequest.generationConfig,
            systemInstruction: geminiRequest.systemInstruction
          });
          const result = await Promise.race([requestPromise, timeoutPromise]);

          if (!result) {
            throw new Error('Empty response from Gemini API');
          }

          const { text, finishReason } = this.extractTextFromCandidates(result);

          if (finishReason === 'MAX_TOKENS') {
            logger.warn('Gemini response reached max tokens limit', {
              attempt,
              finishReason,
              model: modelName
            });
          }

          logger.debug('Gemini API request successful', {
            attempt,
            model: modelName,
            responseLength: text.length,
            finishReason
          });

          return text;
        } catch (error) {
          const errorInfo = this.analyzeError(error);
          lastError = error;

          // Enhanced error logging for fetch failures
          if (errorInfo.type === 'NETWORK_ERROR') {
            logger.error('Network error details', {
              attempt,
              model: modelName,
              errorMessage: error.message,
              errorStack: error.stack,
              errorName: error.name,
              nodeEnv: process.env.NODE_ENV,
              electronVersion: process.versions.electron,
              chromeVersion: process.versions.chrome,
              nodeVersion: process.versions.node,
              userAgent: this.getUserAgent()
            });
          }

          logger.warn(`Gemini API attempt ${attempt} failed for model ${modelName}`, {
            error: error.message,
            errorType: errorInfo.type,
            isNetworkError: errorInfo.isNetworkError,
            suggestedAction: errorInfo.suggestedAction,
            remainingAttempts: maxRetries - attempt,
            model: modelName
          });

          // For model-unavailable / overloaded / rate-limit errors, move to
          // the next fallback model immediately instead of burning all retries.
          const isModelUnavailable = errorInfo.type === 'RATE_LIMIT_ERROR' ||
            error.message.includes('503') ||
            error.message.includes('UNAVAILABLE') ||
            error.message.includes('high demand');

          if (isModelUnavailable && modelName !== modelsToTry[modelsToTry.length - 1]) {
            logger.info(`Switching to fallback model after ${modelName} unavailable`, {
              model: modelName,
              error: error.message
            });
            break; // exit retry loop for this model and try next model
          }

          if (attempt === maxRetries) {
            break; // exit retry loop for this model and try next model
          }

          // Use exponential backoff with jitter for network errors
          const baseDelay = errorInfo.isNetworkError ? 2500 : 1500;
          const delay = baseDelay * attempt + Math.random() * 1000;

          logger.debug(`Waiting ${delay}ms before retry ${attempt + 1}`, {
            baseDelay,
            isNetworkError: errorInfo.isNetworkError,
            model: modelName
          });

          await this.delay(delay);
        }
      }
    }

    const finalErrorInfo = this.analyzeError(lastError);
    const finalError = new Error(`Gemini API failed after trying ${modelsToTry.join(', ')}: ${lastError?.message}`);
    finalError.errorAnalysis = finalErrorInfo;
    finalError.originalError = lastError;
    throw finalError;
  }

  /**
   * Streaming sibling of processTranscriptionWithIntelligentResponse. Emits
   * incremental text via onDelta so the UI can render the answer as it is
   * generated (much faster perceived latency). Returns the same
   * {response, metadata} shape. Falls back to the non-streaming path on any
   * streaming failure so reliability is never worse than before.
   */
  async processTranscriptionWithIntelligentResponseStream(text, activeSkill, sessionMemory = [], programmingLanguage = null, onDelta = null) {
    if (!this.isInitialized) {
      throw new Error(this._notInitializedMessage());
    }

    const startTime = Date.now();
    this.requestCount++;

    try {
      const geminiRequest = this.buildIntelligentTranscriptionRequest(text, activeSkill, sessionMemory, programmingLanguage);

      const fullText = await this.executeStreamingRequest(geminiRequest, (delta) => {
        if (typeof onDelta === 'function' && delta) {
          onDelta(delta);
        }
      });

      const finalResponse = programmingLanguage
        ? this.enforceProgrammingLanguage(fullText, programmingLanguage)
        : fullText;

      logger.logPerformance('LLM transcription streaming', startTime, {
        activeSkill,
        textLength: text.length,
        responseLength: finalResponse.length,
        requestId: this.requestCount
      });

      return {
        response: finalResponse,
        metadata: {
          skill: activeSkill,
          programmingLanguage,
          processingTime: Date.now() - startTime,
          requestId: this.requestCount,
          usedFallback: false,
          streamed: true,
          isTranscriptionResponse: true
        }
      };
    } catch (error) {
      logger.warn('Streaming transcription failed, falling back to non-streaming', {
        error: error.message,
        requestId: this.requestCount
      });
      // Non-streaming path returns the same shape; the caller renders it as a
      // single final response.
      return this.processTranscriptionWithIntelligentResponse(text, activeSkill, sessionMemory, programmingLanguage);
    }
  }

  /** Safely pull the text delta out of a streamed Gemini chunk. */
  _extractChunkText(chunk) {
    try {
      const t = chunk && chunk.text;
      if (typeof t === 'string') {
        return t;
      }
    } catch (_) {
      // `.text` getter can throw on non-text parts; fall through to manual read.
    }
    try {
      const parts = (chunk && chunk.candidates && chunk.candidates[0] &&
        chunk.candidates[0].content && chunk.candidates[0].content.parts) || [];
      return parts.map((p) => (p && typeof p.text === 'string' ? p.text : '')).join('');
    } catch (_) {
      return '';
    }
  }

  // ── OpenRouter (OpenAI-compatible) transport ──────────────────────
  // The rest of the service builds Gemini-shaped request objects. These
  // helpers translate that shape into OpenRouter chat-completions messages
  // and run the HTTPS calls, so every existing process* method works
  // unchanged regardless of the active provider.

  /** Concatenate the text parts of a Gemini `parts` array. */
  _extractGeminiPartsText(parts) {
    if (!Array.isArray(parts)) {
      return '';
    }
    return parts
      .filter((p) => p && typeof p.text === 'string' && p.text.length > 0)
      .map((p) => p.text)
      .join('\n')
      .trim();
  }

  /**
   * Convert a Gemini request object ({contents, systemInstruction,
   * generationConfig}) into an OpenRouter chat-completions payload. Inline
   * image data becomes an OpenAI-style `image_url` data URI, which requires a
   * vision-capable model for screenshot analysis.
   */
  _toOpenRouterPayload(request, model, stream) {
    const messages = [];

    const systemText = this._extractGeminiPartsText(request?.systemInstruction?.parts);
    if (systemText) {
      messages.push({ role: 'system', content: systemText });
    }

    const contents = Array.isArray(request?.contents) ? request.contents : [];
    for (const item of contents) {
      const role = item?.role === 'model' ? 'assistant' : (item?.role || 'user');
      const parts = Array.isArray(item?.parts) ? item.parts : [];
      const hasMedia = parts.some((p) => p && (p.inlineData || p.fileData));

      if (!hasMedia) {
        const text = this._extractGeminiPartsText(parts);
        if (text) {
          messages.push({ role, content: text });
        }
        continue;
      }

      const contentParts = [];
      for (const part of parts) {
        if (part && typeof part.text === 'string' && part.text.length > 0) {
          contentParts.push({ type: 'text', text: part.text });
        } else if (part && part.inlineData) {
          const mime = part.inlineData.mimeType || 'image/png';
          const data = part.inlineData.data || '';
          contentParts.push({
            type: 'image_url',
            image_url: { url: `data:${mime};base64,${data}` }
          });
        }
      }
      if (contentParts.length) {
        messages.push({ role, content: contentParts });
      }
    }

    const generation = request?.generationConfig || {};
    const payload = { model, messages, stream: !!stream };
    if (generation.temperature !== undefined) payload.temperature = generation.temperature;
    if (generation.topP !== undefined) payload.top_p = generation.topP;
    if (generation.topK !== undefined) payload.top_k = generation.topK;
    if (generation.maxOutputTokens !== undefined) payload.max_tokens = generation.maxOutputTokens;
    return payload;
  }

  _openRouterHeaders(apiKey, postData) {
    return {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`,
      'Content-Length': Buffer.byteLength(postData),
      // App attribution headers (optional, but requested by OpenRouter docs).
      'HTTP-Referer': 'https://github.com/TechyCSR/OpenCluely',
      'X-OpenRouter-Title': 'OpenCluely',
      'User-Agent': this.getUserAgent()
    };
  }

  /** Non-streaming OpenRouter call with model-fallback + retry. */
  async _executeOpenRouterRequest(request) {
    const maxRetries = config.get('llm.openrouter.maxRetries') || 3;
    const fallbackModels = config.get('llm.openrouter.fallbackModels') || [];
    const modelsToTry = [this.model, ...fallbackModels].filter(Boolean);
    let lastError = null;

    for (const modelName of modelsToTry) {
      for (let attempt = 1; attempt <= maxRetries; attempt++) {
        try {
          return await this._openRouterNonStreamingCall(request, modelName);
        } catch (error) {
          const errorInfo = this.analyzeError(error);
          lastError = error;

          logger.warn(`OpenRouter attempt ${attempt} failed for model ${modelName}`, {
            error: error.message,
            errorType: errorInfo.type,
            model: modelName
          });

          const isModelUnavailable = errorInfo.type === 'RATE_LIMIT_ERROR' ||
            error.message.includes('503') ||
            error.message.includes('404') ||
            error.message.includes('unavailable');

          if (isModelUnavailable && modelName !== modelsToTry[modelsToTry.length - 1]) {
            break; // try the next fallback model
          }
          if (attempt === maxRetries) {
            break;
          }

          const baseDelay = errorInfo.isNetworkError ? 2500 : 1500;
          await this.delay(baseDelay * attempt + Math.random() * 1000);
        }
      }
    }

    const finalError = new Error(`OpenRouter API failed: ${lastError?.message}`);
    finalError.originalError = lastError;
    throw finalError;
  }

  _openRouterNonStreamingCall(request, modelName) {
    const https = require('https');
    const apiKey = config.getApiKey('OPENROUTER');
    const timeout = config.get('llm.openrouter.timeout') || 60000;
    const postData = JSON.stringify(this._toOpenRouterPayload(request, modelName, false));
    const agent = new https.Agent({ keepAlive: true, maxSockets: 1 });

    const options = {
      hostname: 'openrouter.ai',
      path: '/api/v1/chat/completions',
      method: 'POST',
      headers: this._openRouterHeaders(apiKey, postData),
      timeout,
      agent
    };

    return new Promise((resolve, reject) => {
      const req = https.request(options, (res) => {
        let data = '';
        res.setEncoding('utf8');
        res.on('data', (chunk) => { data += chunk; });
        res.on('end', () => {
          if (res.statusCode < 200 || res.statusCode >= 300) {
            reject(new Error(`HTTP ${res.statusCode}: ${data.substring(0, 500)}`));
            return;
          }
          try {
            const json = JSON.parse(data);
            if (json && json.error) {
              reject(new Error(json.error.message || 'OpenRouter error'));
              return;
            }
            const content = json?.choices?.[0]?.message?.content;
            if (typeof content !== 'string' || !content.trim()) {
              reject(new Error('Empty response from OpenRouter'));
              return;
            }
            resolve(content.trim());
          } catch (parseError) {
            reject(new Error(`Failed to parse OpenRouter response: ${parseError.message}`));
          }
        });
      });

      req.on('error', (error) => reject(new Error(`OpenRouter request failed: ${error.message}`)));
      req.on('timeout', () => {
        req.destroy();
        reject(new Error('OpenRouter request timeout'));
      });

      req.write(postData);
      req.end();
    });
  }

  /** Pull the text delta out of a streamed OpenRouter/OpenAI chunk. */
  _extractOpenRouterChunkText(chunk) {
    try {
      const choice = chunk && chunk.choices && chunk.choices[0];
      const delta = choice && (choice.delta || choice.message);
      if (delta && typeof delta.content === 'string') {
        return delta.content;
      }
    } catch (_) {
      // ignore malformed chunks
    }
    return '';
  }

  _streamOpenRouterRequest(request, modelName, onDelta) {
    const https = require('https');
    const apiKey = config.getApiKey('OPENROUTER');
    const timeout = config.get('llm.openrouter.timeout') || 60000;
    const postData = JSON.stringify(this._toOpenRouterPayload(request, modelName, true));
    const agent = new https.Agent({ keepAlive: true, maxSockets: 1 });

    const options = {
      hostname: 'openrouter.ai',
      path: '/api/v1/chat/completions',
      method: 'POST',
      headers: {
        ...this._openRouterHeaders(apiKey, postData),
        'Accept': 'text/event-stream'
      },
      timeout,
      agent
    };

    return new Promise((resolve, reject) => {
      const req = https.request(options, (res) => {
        if (res.statusCode !== 200) {
          let errBody = '';
          res.on('data', (c) => { errBody += c; });
          res.on('end', () => reject(new Error(`HTTP ${res.statusCode}: ${errBody.substring(0, 500)}`)));
          return;
        }

        let fullText = '';
        let buffer = '';

        res.setEncoding('utf8');
        res.on('data', (chunk) => {
          buffer += chunk;
          let idx;
          while ((idx = buffer.indexOf('\n')) !== -1) {
            const line = buffer.slice(0, idx).trim();
            buffer = buffer.slice(idx + 1);
            if (!line.startsWith('data:')) {
              continue;
            }
            const payload = line.slice(5).trim();
            if (!payload || payload === '[DONE]') {
              continue;
            }
            try {
              const json = JSON.parse(payload);
              const piece = this._extractOpenRouterChunkText(json);
              if (piece) {
                fullText += piece;
                if (typeof onDelta === 'function') {
                  onDelta(piece);
                }
              }
            } catch (_) {
              // Partial JSON across chunk boundaries; skip unparseable lines.
            }
          }
        });

        res.on('end', () => resolve(fullText.trim()));
        res.on('error', (error) => reject(new Error(`OpenRouter stream error: ${error.message}`)));
      });

      req.on('error', (error) => reject(new Error(`OpenRouter streaming request failed: ${error.message}`)));
      req.on('timeout', () => {
        req.destroy();
        reject(new Error('OpenRouter streaming request timeout'));
      });

      req.write(postData);
      req.end();
    });
  }

  /**
   * Run a streaming Gemini request with the same model-fallback + retry policy
   * as executeRequest. Accumulates and returns the full text; invokes onDelta
   * for each chunk.
   */
  async executeStreamingRequest(geminiRequest, onDelta) {
    const isOpenRouter = this.provider === 'openrouter';
    const maxRetries = config.get(isOpenRouter ? 'llm.openrouter.maxRetries' : 'llm.gemini.maxRetries');
    const apiKey = isOpenRouter ? config.getApiKey('OPENROUTER') : config.getApiKey('GEMINI');
    const primaryModel = this.model;
    const fallbackModels = config.get(isOpenRouter ? 'llm.openrouter.fallbackModels' : 'llm.gemini.fallbackModels') || [];
    const modelsToTry = [primaryModel, ...fallbackModels];

    let lastError = null;

    for (const modelName of modelsToTry) {
      for (let attempt = 1; attempt <= maxRetries; attempt++) {
        try {
          const fullText = await this._streamRequestForModel(geminiRequest, modelName, apiKey, onDelta);

          if (!fullText) {
            throw new Error('Empty streamed response from Gemini API');
          }

          logger.debug('Gemini streaming request successful', {
            attempt,
            model: modelName,
            responseLength: fullText.length
          });

          return fullText;
        } catch (error) {
          const errorInfo = this.analyzeError(error);
          lastError = error;

          logger.warn(`Gemini streaming attempt ${attempt} failed for model ${modelName}`, {
            error: error.message,
            errorType: errorInfo.type,
            remainingAttempts: maxRetries - attempt,
            model: modelName
          });

          const isModelUnavailable = errorInfo.type === 'RATE_LIMIT_ERROR' ||
            error.message.includes('503') ||
            error.message.includes('UNAVAILABLE') ||
            error.message.includes('high demand');

          if (isModelUnavailable && modelName !== modelsToTry[modelsToTry.length - 1]) {
            break; // try next fallback model
          }

          if (attempt === maxRetries) {
            break;
          }

          const baseDelay = errorInfo.isNetworkError ? 2500 : 1500;
          const delay = baseDelay * attempt + Math.random() * 1000;
          await this.delay(delay);
        }
      }
    }

    throw lastError || new Error('Gemini streaming request failed');
  }

  _streamRequestForModel(geminiRequest, modelName, apiKey, onDelta) {
    if (this.provider === 'openrouter') {
      return this._streamOpenRouterRequest(geminiRequest, modelName, onDelta);
    }

    const https = require('https');
    const timeout = config.get('llm.gemini.timeout');
    const url = `https://generativelanguage.googleapis.com/v1beta/models/${modelName}:streamGenerateContent?alt=sse`;
    const postData = JSON.stringify(geminiRequest);
    const agent = new https.Agent({ keepAlive: true, maxSockets: 1 });

    const options = {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-goog-api-key': apiKey,
        'Content-Length': Buffer.byteLength(postData),
        'User-Agent': this.getUserAgent()
      },
      timeout,
      agent
    };

    return new Promise((resolve, reject) => {
      const req = https.request(url, options, (res) => {
        if (res.statusCode !== 200) {
          let errBody = '';
          res.on('data', (c) => { errBody += c; });
          res.on('end', () => reject(new Error(`HTTP ${res.statusCode}: ${errBody}`)));
          return;
        }

        let fullText = '';
        let buffer = '';

        res.setEncoding('utf8');
        res.on('data', (chunk) => {
          buffer += chunk;
          let idx;
          while ((idx = buffer.indexOf('\n')) !== -1) {
            const line = buffer.slice(0, idx).trim();
            buffer = buffer.slice(idx + 1);
            if (!line.startsWith('data:')) {
              continue;
            }
            const payload = line.slice(5).trim();
            if (!payload || payload === '[DONE]') {
              continue;
            }
            try {
              const json = JSON.parse(payload);
              const piece = this._extractChunkText(json);
              if (piece) {
                fullText += piece;
                if (typeof onDelta === 'function') {
                  onDelta(piece);
                }
              }
            } catch (_) {
              // Partial JSON across chunk boundaries is rare with line framing;
              // skip anything that doesn't parse cleanly.
            }
          }
        });

        res.on('end', () => resolve(fullText.trim()));
        res.on('error', (error) => reject(new Error(`Streaming response error: ${error.message}`)));
      });

      req.on('error', (error) => reject(new Error(`Streaming request failed: ${error.message}`)));
      req.on('timeout', () => {
        req.destroy();
        reject(new Error('Streaming request timeout'));
      });

      req.write(postData);
      req.end();
    });
  }

  async performPreflightCheck() {
    // Quick connectivity check
    try {
      const startTime = Date.now();
      await this.testNetworkConnection({ 
        host: 'generativelanguage.googleapis.com', 
        port: 443, 
        name: 'Gemini API Endpoint' 
      });
      const latency = Date.now() - startTime;
      
      logger.debug('Preflight check passed', { latency });
    } catch (error) {
      logger.warn('Preflight check failed', { 
        error: error.message,
        suggestion: 'Network connectivity issue detected before API call'
      });
      // Don't throw here - let the actual API call fail with more detail
    }
  }

  getUserAgent() {
    try {
      // Try to get user agent from Electron if available
      if (typeof navigator !== 'undefined' && navigator.userAgent) {
        return navigator.userAgent;
      }
      return `Node.js/${process.version} (${process.platform}; ${process.arch})`;
    } catch {
      return 'Unknown';
    }
  }

  analyzeError(error) {
    const errorMessage = error.message.toLowerCase();
    
    // Network connectivity errors
    if (errorMessage.includes('fetch failed') || 
        errorMessage.includes('network error') ||
        errorMessage.includes('enotfound') ||
        errorMessage.includes('econnrefused') ||
        errorMessage.includes('timeout')) {
      return {
        type: 'NETWORK_ERROR',
        isNetworkError: true,
        suggestedAction: 'Check internet connection and firewall settings'
      };
    }
    
    // API key errors
    if (errorMessage.includes('unauthorized') || 
        errorMessage.includes('invalid api key') ||
        errorMessage.includes('no auth credentials') ||
        errorMessage.includes('forbidden') ||
        errorMessage.includes('http 401') ||
        errorMessage.includes('http 403')) {
      return {
        type: 'AUTH_ERROR',
        isNetworkError: false,
        suggestedAction: 'Verify the configured API key'
      };
    }
    
    // Rate limiting / insufficient credits
    if (errorMessage.includes('quota') || 
        errorMessage.includes('rate limit') ||
        errorMessage.includes('too many requests') ||
        errorMessage.includes('http 429') ||
        errorMessage.includes('http 402') ||
        errorMessage.includes('insufficient credits')) {
      return {
        type: 'RATE_LIMIT_ERROR',
        isNetworkError: false,
        suggestedAction: 'Wait before retrying or check API quota'
      };
    }
    
    // Timeout errors
    if (errorMessage.includes('request timeout') || errorMessage.includes('etimedout')) {
      return {
        type: 'TIMEOUT_ERROR',
        isNetworkError: true,
        suggestedAction: 'Check network latency or increase timeout'
      };
    }
    
    return {
      type: 'UNKNOWN_ERROR',
      isNetworkError: false,
      suggestedAction: 'Check logs for more details'
    };
  }

  async checkNetworkConnectivity() {
    const connectivityTests = [
      { host: 'google.com', port: 443, name: 'Google (HTTPS)' },
      { host: 'generativelanguage.googleapis.com', port: 443, name: 'Gemini API Endpoint' },
      { host: 'openrouter.ai', port: 443, name: 'OpenRouter API Endpoint' }
    ];

    const results = await Promise.allSettled(
      connectivityTests.map(test => this.testNetworkConnection(test))
    );

    const connectivity = {
      timestamp: new Date().toISOString(),
      tests: results.map((result, index) => ({
        ...connectivityTests[index],
        success: result.status === 'fulfilled' && result.value,
        error: result.status === 'rejected' ? result.reason.message : null
      }))
    };

    logger.info('Network connectivity check completed', connectivity);
    return connectivity;
  }

  async testNetworkConnection({ host, port, name }) {
    return new Promise((resolve, reject) => {
      const net = require('net');
      const socket = new net.Socket();
      
      const timeout = setTimeout(() => {
        socket.destroy();
        reject(new Error(`Connection timeout to ${host}:${port}`));
      }, 5000);

      socket.on('connect', () => {
        clearTimeout(timeout);
        socket.destroy();
        resolve(true);
      });

      socket.on('error', (error) => {
        clearTimeout(timeout);
        reject(new Error(`Connection failed to ${host}:${port}: ${error.message}`));
      });

      socket.connect(port, host);
    });
  }

  generateFallbackResponse(text, activeSkill) {
    logger.info('Generating fallback response', { activeSkill });

    const fallbackResponses = {
      'interview': 'I could not reach the model for that one. Give it another try in a moment.',
      'dsa': 'This appears to be a data structures and algorithms problem. Consider breaking it down into smaller components and identifying the appropriate algorithm or data structure to use.',
      'system-design': 'For this system design question, consider scalability, reliability, and the trade-offs between different architectural approaches.',
      'programming': 'This looks like a programming challenge. Focus on understanding the requirements, edge cases, and optimal time/space complexity.',
      'default': 'I can help analyze this content. Please ensure your AI provider API key is properly configured for detailed analysis.'
    };

    const response = fallbackResponses[activeSkill] || fallbackResponses.default;
    
    return {
      response,
      metadata: {
        skill: activeSkill,
        processingTime: 0,
        requestId: this.requestCount,
        usedFallback: true
      }
    };
  }

  generateIntelligentFallbackResponse(text, activeSkill) {
    logger.info('Generating intelligent fallback response for transcription', { activeSkill });

    // Keep the fallback short and neutral: the candidate reads this live, so a
    // long apology or a skill mismatch would be worse than a one-liner.
    return {
      response: "I couldn't reach the model for that just now — give it another try in a moment.",
      metadata: {
        skill: activeSkill,
        processingTime: 0,
        requestId: this.requestCount,
        usedFallback: true,
        isTranscriptionResponse: true
      }
    };
  }

  async testConnection() {
    if (!this.isInitialized) {
      return { success: false, error: 'Service not initialized' };
    }

    if (this.provider === 'openrouter') {
      return this._testOpenRouterConnection();
    }

    try {
      // First check network connectivity
      const networkCheck = await this.checkNetworkConnectivity();
      const hasNetworkIssues = networkCheck.tests.some(test => !test.success);
      
      if (hasNetworkIssues) {
        logger.warn('Network connectivity issues detected', networkCheck);
      }

      const generationConfig = this.getGenerationConfig({ temperature: 0, maxOutputTokens: 64 });
      const fallbackModels = config.get('llm.gemini.fallbackModels') || [];
      const modelsToTry = [this.model, ...fallbackModels];

      let lastError = null;
      let result = null;
      let usedModel = null;

      for (const modelName of modelsToTry) {
        try {
          const startTime = Date.now();
          result = await this.client.models.generateContent({
            model: modelName,
            contents: 'Test connection. Please respond with "OK".',
            config: generationConfig
          });
          usedModel = modelName;
          const latency = Date.now() - startTime;
          const { text } = this.extractTextFromCandidates(result);

          logger.info('Connection test successful', {
            response: text,
            latency,
            model: usedModel,
            networkCheck: hasNetworkIssues ? 'issues_detected' : 'healthy'
          });

          return {
            success: true,
            response: text,
            latency,
            model: usedModel,
            networkConnectivity: networkCheck
          };
        } catch (error) {
          lastError = error;
          logger.warn(`Connection test failed for model ${modelName}`, {
            error: error.message,
            model: modelName
          });

          const isModelUnavailable = error.message.includes('503') ||
            error.message.includes('UNAVAILABLE') ||
            error.message.includes('high demand') ||
            error.message.includes('quota') ||
            error.message.includes('rate limit');

          if (!isModelUnavailable && modelName === this.model) {
            // Primary model failed for a non-availability reason; don't hide it
            break;
          }
        }
      }

      throw lastError || new Error('Connection test failed on all models');
    } catch (error) {
      const errorAnalysis = this.analyzeError(error);
      logger.error('Connection test failed', {
        error: error.message,
        errorAnalysis
      });

      // Map raw SDK errors to user-friendly messages. The wizard only
      // surfaces `error`, so any raw SDK error string would land in the
      // UI verbatim.
      const friendlyError = this._friendlyTestError(error, errorAnalysis);

      return {
        success: false,
        error: friendlyError,
        errorType: errorAnalysis?.type || 'UNKNOWN',
        errorAnalysis,
        networkConnectivity: await this.checkNetworkConnectivity().catch(() => null)
      };
    }
  }

  /**
   * OpenRouter connection test: a minimal chat completion against the
   * configured model, mirroring the Gemini test's {success, latency, model}
   * shape so callers/UI work unchanged.
   */
  async _testOpenRouterConnection() {
    const networkCheck = await this.checkNetworkConnectivity().catch(() => null);
    const startTime = Date.now();

    try {
      const text = await this._openRouterNonStreamingCall({
        contents: [{ role: 'user', parts: [{ text: 'Test connection. Please respond with "OK".' }] }],
        generationConfig: { temperature: 0, maxOutputTokens: 16 }
      }, this.model);

      return {
        success: true,
        response: text,
        latency: Date.now() - startTime,
        model: this.model,
        networkConnectivity: networkCheck
      };
    } catch (error) {
      const errorAnalysis = this.analyzeError(error);
      logger.error('OpenRouter connection test failed', {
        error: error.message,
        errorAnalysis
      });
      return {
        success: false,
        error: this._friendlyTestError(error, errorAnalysis),
        errorType: errorAnalysis?.type || 'UNKNOWN',
        errorAnalysis,
        networkConnectivity: networkCheck
      };
    }
  }

  /**
   * Translate raw SDK / network errors into something a user can act on.
   */
  _friendlyTestError(error, analysis) {
    const type = analysis?.type;
    const raw = (error?.message || '').toLowerCase();
    const isOpenRouter = this.provider === 'openrouter';
    const hostLabel = isOpenRouter ? 'OpenRouter' : 'Google';
    const keyHint = isOpenRouter
      ? 'Double-check the key at openrouter.ai/settings/keys.'
      : 'Double-check the key at aistudio.google.com/apikey.';

    if (type === 'NETWORK_ERROR' || raw.includes('fetch failed') || raw.includes('enotfound')) {
      return `Cannot reach ${hostLabel} servers. Check your internet connection, firewall, or VPN settings.`;
    }
    if (type === 'AUTH_ERROR' || raw.includes('api key') || raw.includes('401') || raw.includes('403')) {
      return `Invalid API key or insufficient permissions. ${keyHint}`;
    }
    if (type === 'RATE_LIMIT_ERROR' || raw.includes('429') || raw.includes('quota') || raw.includes('402')) {
      return isOpenRouter
        ? 'Rate limit or insufficient credits on OpenRouter. Check your balance or wait a moment.'
        : 'Rate limit or quota exceeded. Wait a moment or check your Google Cloud billing.';
    }
    if (type === 'TIMEOUT_ERROR') {
      return `Request timed out. The ${hostLabel} API may be slow or unreachable right now.`;
    }
    if (type === 'MODEL_ERROR' || raw.includes('404')) {
      return isOpenRouter
        ? 'The configured OpenRouter model is unavailable. Try a different model slug in Settings.'
        : 'The configured Gemini model is unavailable. Try a different model in Settings.';
    }
    if (raw.includes('503') || raw.includes('unavailable') || raw.includes('high demand')) {
      return `${hostLabel} is experiencing high demand. Please wait a moment and try again.`;
    }
    // Fall back to a stripped-down raw message (no SDK prefix noise)
    return (error?.message || 'Connection failed').replace(/^\[(GoogleGenerativeAI|GoogleGenAI) Error\]:\s*/i, '');
  }

  updateApiKey(newApiKey, provider = this.provider) {
    if (provider === 'openrouter') {
      process.env.OPENROUTER_API_KEY = newApiKey;
    } else {
      process.env.GEMINI_API_KEY = newApiKey;
    }
    this.isInitialized = false;
    this.initializeClient();
    
    logger.info('API key updated and client reinitialized', { provider });
  }

  getStats() {
    return {
      provider: this.provider,
      model: this.model,
      isInitialized: this.isInitialized,
      requestCount: this.requestCount,
      errorCount: this.errorCount,
      successRate: this.requestCount > 0 ? ((this.requestCount - this.errorCount) / this.requestCount) * 100 : 0,
      config: this.provider === 'openrouter'
        ? config.get('llm.openrouter')
        : config.get('llm.gemini')
    };
  }

  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  async executeAlternativeRequest(geminiRequest) {
    // OpenRouter has a single transport, so the "alternative" method just
    // retries the same OpenRouter request. This keeps the existing
    // primary→secondary fallback logic in the process* methods intact.
    if (this.provider === 'openrouter') {
      return this._executeOpenRouterRequest(geminiRequest);
    }

    const https = require('https');
    const apiKey = config.getApiKey('GEMINI');
    const primaryModel = config.get('llm.gemini.model');
    const fallbackModels = config.get('llm.gemini.fallbackModels') || [];
    const modelsToTry = [primaryModel, ...fallbackModels];

    logger.info('Using alternative HTTPS request method', { modelsToTry });

    let lastError = null;

    for (const modelName of modelsToTry) {
      try {
        const result = await this._executeAlternativeRequestForModel(geminiRequest, modelName, apiKey);
        return result;
      } catch (error) {
        lastError = error;
        logger.warn(`Alternative HTTPS request failed for model ${modelName}`, {
          error: error.message,
          model: modelName
        });

        const isModelUnavailable = error.message.includes('503') ||
          error.message.includes('UNAVAILABLE') ||
          error.message.includes('high demand');

        if (!isModelUnavailable && modelName === primaryModel) {
          break;
        }
      }
    }

    throw lastError || new Error('Alternative HTTPS request failed for all models');
  }

  async _executeAlternativeRequestForModel(geminiRequest, modelName, apiKey) {
    const https = require('https');

    const url = `https://generativelanguage.googleapis.com/v1beta/models/${modelName}:generateContent`;

    const postData = JSON.stringify(geminiRequest);

    const agent = new https.Agent({ keepAlive: true, maxSockets: 1 });

    const options = {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-goog-api-key': apiKey,
        'Content-Length': Buffer.byteLength(postData),
        'User-Agent': this.getUserAgent()
      },
      timeout: config.get('llm.gemini.timeout'),
      agent
    };

    return new Promise((resolve, reject) => {
      const req = https.request(url, options, (res) => {
        let data = '';
        
        res.on('data', (chunk) => {
          data += chunk;
        });
        
        res.on('end', () => {
          try {
            if (res.statusCode !== 200) {
              reject(new Error(`HTTP ${res.statusCode}: ${data}`));
              return;
            }
            
            const response = JSON.parse(data);
            
            logger.debug('Alternative request response structure', {
              hasResponse: !!response,
              hasCandidates: !!response.candidates,
              candidatesLength: response.candidates?.length,
              responseKeys: Object.keys(response || {}),
              firstCandidateKeys: response.candidates?.[0] ? Object.keys(response.candidates[0]) : []
            });

            const { text, finishReason } = this.extractTextFromCandidates(response);

            if (finishReason === 'MAX_TOKENS') {
              logger.warn('Gemini alternative response reached max tokens limit', {
                finishReason
              });
            }
            
            logger.info('Alternative request successful', {
              responseLength: text.length,
              statusCode: res.statusCode,
              finishReason
            });
            
            resolve(text.trim());
          } catch (parseError) {
            logger.error('Failed to parse alternative response', {
              error: parseError.message,
              rawResponse: data.substring(0, 500),
              statusCode: res.statusCode
            });
            reject(new Error(`Failed to parse response: ${parseError.message}`));
          }
        });
      });
      
      req.on('error', (error) => {
        reject(new Error(`Alternative request failed: ${error.message}`));
      });
      
      req.on('timeout', () => {
        req.destroy();
        reject(new Error('Alternative request timeout'));
      });
      
      req.write(postData);
      req.end();
    });
  }
}

module.exports = new LLMService();