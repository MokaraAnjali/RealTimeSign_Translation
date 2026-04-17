import React, { useEffect, useRef, useState } from 'react';
import { Mic, MicOff, Send, Volume2, Languages, Settings, Sparkles, Video, VideoOff, AlertCircle, Camera, Square, CircleDot } from 'lucide-react';
import SpeechRecognition, { useSpeechRecognition } from 'react-speech-recognition';
import axios from 'axios';
import Webcam from 'react-webcam';
import './ChatTranslation.css';

const tones = [
  { label: 'Professional', icon: '💼', description: 'Formal and respectful' },
  { label: 'Casual', icon: '👋', description: 'Friendly and relaxed' },
  { label: 'Empathetic', icon: '❤️', description: 'Kind and supportive' },
  { label: 'Inclusive', icon: '🌈', description: 'Gender-neutral and respectful' },
];

const languages = [
  { code: 'en-US', name: 'English', shortCode: 'en', flag: '🇺🇸' },
  { code: 'es-ES', name: 'Spanish', shortCode: 'es', flag: '🇪🇸' },
  { code: 'fr-FR', name: 'French', shortCode: 'fr', flag: '🇫🇷' },
  { code: 'de-DE', name: 'German', shortCode: 'de', flag: '🇩🇪' },
  { code: 'hi-IN', name: 'Hindi', shortCode: 'hi', flag: '🇮🇳' },
  { code: 'ta-IN', name: 'Tamil', shortCode: 'ta', flag: '🇮🇳' },
  { code: 'te-IN', name: 'Telugu', shortCode: 'te', flag: '🇮🇳' },
  { code: 'ja-JP', name: 'Japanese', shortCode: 'ja', flag: '🇯🇵' },
];

const BACKEND_TTS_LANGUAGES = new Set(['hi', 'ta', 'te']);
const EMOJI_PATTERN = /[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}]/gu;
const SARCASM_MARKERS = ['yeah right', 'sure', 'totally', 'obviously', '/s', 'as if', 'great...', 'fine 🙂', 'fine 🙃'];
const HEDGING_MARKERS = ['maybe', 'perhaps', 'possibly', 'i think', 'i feel', 'kind of', 'sort of', 'could we'];

const EMOTION_RULES = [
  { label: 'Happy', icon: '😊', words: ['happy', 'great', 'good', 'awesome', 'glad', 'excited', 'love'] },
  { label: 'Empathetic', icon: '😔', words: ['sad', 'bad', 'sorry', 'hurt', 'upset', 'stress', 'worried'] },
  { label: 'Urgent', icon: '⚠️', words: ['urgent', 'help', 'now', 'asap', 'immediately', 'emergency'] },
  { label: 'Curious', icon: '🤔', words: ['how', 'why', 'what', 'can', 'could', '?'] },
];

const TONE_HINTS = {
  Empathetic: ['sad', 'sorry', 'difficult', 'hurt', 'lonely', 'worried', 'stress'],
  Casual: ['hey', 'hi', 'hello', 'thanks', 'cool', 'awesome', 'lol'],
  Professional: ['please', 'kindly', 'regarding', 'request', 'schedule', 'meeting'],
};

const getLanguageMeta = (code) => languages.find((language) => language.code === code) || languages[0];

const detectEmotion = (text) => {
  const normalizedText = text.toLowerCase();
  const match = EMOTION_RULES.find(({ words }) =>
    words.some((word) => normalizedText.includes(word))
  );

  return match ? `${match.label} ${match.icon}` : 'Neutral 😐';
};

const inferTone = (text, fallbackTone) => {
  const normalizedText = text.toLowerCase();

  for (const [tone, keywords] of Object.entries(TONE_HINTS)) {
    if (keywords.some((word) => normalizedText.includes(word))) {
      return tone;
    }
  }

  if (normalizedText.includes('?')) {
    return 'Professional';
  }

  return fallbackTone;
};

const extractProcessedResponse = (data, originalText) => {
  if (typeof data === 'string') {
    return { translatedText: data, detectedTone: null, detectedEmotion: null, method: null };
  }

  if (!data || typeof data !== 'object') {
    return { translatedText: originalText, detectedTone: null, detectedEmotion: null, method: null };
  }

  return {
    translatedText:
      data.processed_text ||
      data.translated_text ||
      data.translation ||
      data.response ||
      originalText,
    detectedTone: data.detected_tone || data.tone || null,
    detectedEmotion: data.detected_emotion || data.emotion || null,
    method: data.method || null,
    message: data.message || null,
    appliedControls: data.applied_controls || null,
  };
};

const looksTranslated = (text, targetLanguage) => {
  if (!text) {
    return false;
  }

  const normalizedText = text.toLowerCase();
  if (targetLanguage.shortCode === 'hi') {
    return /[\u0900-\u097f]/.test(text);
  }

  if (targetLanguage.shortCode === 'ta') {
    return /[\u0B80-\u0BFF]/.test(text);
  }

  if (targetLanguage.shortCode === 'te') {
    return /[\u0C00-\u0C7F]/.test(text);
  }

  if (targetLanguage.shortCode === 'ja') {
    return /[\u3040-\u30ff\u4e00-\u9faf]/.test(text);
  }

  if (targetLanguage.shortCode === 'es') {
    return /[áéíóúñ¿¡]/i.test(text) || normalizedText.includes('hola');
  }

  if (targetLanguage.shortCode === 'fr') {
    return /[àâçéèêëîïôûùüÿœ]/i.test(text) || normalizedText.includes('bonjour');
  }

  if (targetLanguage.shortCode === 'de') {
    return /[äöüß]/i.test(text) || normalizedText.includes('hallo');
  }

  return normalizedText !== '';
};

const detectSarcasm = (text) => {
  const normalizedText = text.toLowerCase();
  return SARCASM_MARKERS.some((marker) => normalizedText.includes(marker)) || text.includes('!?') || text.includes('?!') || text.includes('🙃');
};

const analyzeParalinguistics = (text) => {
  const emojiMatches = text.match(EMOJI_PATTERN) || [];
  const uppercaseLetters = (text.match(/[A-Z]/g) || []).length;
  const alphabeticLetters = (text.match(/[A-Za-z]/g) || []).length;
  const punctuationBurst = (text.match(/[!?]{2,}|\.{3,}/g) || []).length;

  return {
    emoji_count: emojiMatches.length,
    punctuation_burst: punctuationBurst,
    uppercase_ratio: alphabeticLetters ? Number((uppercaseLetters / alphabeticLetters).toFixed(2)) : 0,
    passive_aggressive_hint: /(fine|ok|okay)\s*[🙂🙃]/iu.test(text),
  };
};

const detectCodeSwitching = (text) => {
  const hasLatin = /[A-Za-z]/.test(text);
  const hasHindi = /[\u0900-\u097f]/.test(text);
  const hasTamil = /[\u0B80-\u0BFF]/.test(text);
  const hasTelugu = /[\u0C00-\u0C7F]/.test(text);

  const scriptCount = [hasLatin, hasHindi, hasTamil, hasTelugu].filter(Boolean).length;
  return {
    detected: scriptCount > 1,
    script_mix_count: scriptCount,
  };
};

const inferRelationshipState = (messages) => {
  const userMessages = messages.filter((message) => message.sender === 'user');
  const historyText = userMessages.map((message) => message.text.toLowerCase()).join(' ');

  if (userMessages.length >= 8 || /(buddy|friend|bro|sis|haha|lol)/.test(historyText)) {
    return 'friend';
  }

  if (/(meeting|review|project|deadline|sir|madam|team)/.test(historyText)) {
    return 'colleague';
  }

  return 'stranger';
};

const getAssertivenessLabel = (value) => {
  if (value <= -2) return 'Soften';
  if (value === -1) return 'Polite';
  if (value === 0) return 'Balanced';
  if (value === 1) return 'Confident';
  return 'Assertive';
};

const computeCognitiveLoad = (telemetry, text) => {
  const averageInterval = telemetry.keyIntervals.length
    ? telemetry.keyIntervals.reduce((sum, interval) => sum + interval, 0) / telemetry.keyIntervals.length
    : 0;

  const longPauseCount = telemetry.keyIntervals.filter((interval) => interval > 1800).length;
  const errorRate = telemetry.charactersTyped
    ? telemetry.backspaceCount / telemetry.charactersTyped
    : 0;

  const highLoad =
    averageInterval > 900 ||
    errorRate > 0.14 ||
    longPauseCount >= 2 ||
    text.length < 12;

  const mediumLoad =
    averageInterval > 500 ||
    errorRate > 0.07 ||
    longPauseCount >= 1;

  if (highLoad) return 'high';
  if (mediumLoad) return 'medium';
  return 'low';
};

const buildConversationSignals = ({ text, messages, telemetry, assertivenessLevel }) => {
  const paralinguistics = analyzeParalinguistics(text);
  const codeSwitching = detectCodeSwitching(text);
  const cognitiveLoad = computeCognitiveLoad(telemetry, text);
  const relationshipState = inferRelationshipState(messages);
  const sarcasmDetected = detectSarcasm(text) || paralinguistics.passive_aggressive_hint;
  const negotiationPower = HEDGING_MARKERS.some((marker) => text.toLowerCase().includes(marker))
    ? 'hedged'
    : 'neutral';

  return {
    cognitive_load: cognitiveLoad,
    sarcasm_detected: sarcasmDetected,
    relationship_state: relationshipState,
    code_switching_detected: codeSwitching.detected,
    script_mix_count: codeSwitching.script_mix_count,
    paralinguistic_signals: paralinguistics,
    negotiation_power_signal: negotiationPower,
    assertiveness_level: assertivenessLevel,
    simplification_preference: cognitiveLoad === 'high' ? 'high' : cognitiveLoad === 'medium' ? 'medium' : 'low',
  };
};

const ChatTranslation = () => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      text: 'Hello! How can I help you today?',
      translatedText: 'Hola! Como puedo ayudarte hoy?',
      sender: 'bot',
      emotion: 'Friendly 😊',
      originalLang: 'en-US',
      targetLang: 'es-ES',
      tone: 'Professional',
      inputMode: 'system',
    },
  ]);
  const [inputText, setInputText] = useState('');
  const [sourceLang, setSourceLang] = useState('en-US');
  const [targetLang, setTargetLang] = useState('es-ES');
  const [isTranslating, setIsTranslating] = useState(false);
  const [selectedTone, setSelectedTone] = useState('Professional');
  const [detectedEmotion, setDetectedEmotion] = useState('Neutral 😐');
  const [showSettings, setShowSettings] = useState(false);
  const [isDetectingSign, setIsDetectingSign] = useState(false);
  const [detectionLang, setDetectionLang] = useState('ISL');
  const [chatError, setChatError] = useState('');
  const [meetingMode, setMeetingMode] = useState(false);
  const [meetingTranscript, setMeetingTranscript] = useState([]);
  const [meetingStatus, setMeetingStatus] = useState('Idle');
  const [isCameraEnabled, setIsCameraEnabled] = useState(false);
  const [availableVoices, setAvailableVoices] = useState([]);
  const [assertivenessLevel, setAssertivenessLevel] = useState(0);
  const [conversationSignals, setConversationSignals] = useState({
    cognitive_load: 'low',
    sarcasm_detected: false,
    relationship_state: 'stranger',
    code_switching_detected: false,
    script_mix_count: 1,
    paralinguistic_signals: {
      emoji_count: 0,
      punctuation_burst: 0,
      uppercase_ratio: 0,
      passive_aggressive_hint: false,
    },
    negotiation_power_signal: 'neutral',
    assertiveness_level: 0,
    simplification_preference: 'low',
  });

  const messagesEndRef = useRef(null);
  const webcamRef = useRef(null);
  const audioRef = useRef(null);
  const audioUrlRef = useRef(null);
  const typingSessionRef = useRef({
    lastTimestamp: null,
    keyIntervals: [],
    backspaceCount: 0,
    charactersTyped: 0,
  });
  const { transcript, listening, resetTranscript, browserSupportsSpeechRecognition } = useSpeechRecognition();

  useEffect(() => {
    const updateVoices = () => {
      setAvailableVoices(window.speechSynthesis.getVoices());
    };

    updateVoices();
    window.speechSynthesis.onvoiceschanged = updateVoices;

    return () => {
      window.speechSynthesis.onvoiceschanged = null;
    };
  }, []);

  useEffect(() => () => {
    if (audioRef.current) {
      audioRef.current.pause();
    }
    if (audioUrlRef.current) {
      URL.revokeObjectURL(audioUrlRef.current);
    }
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTranslating]);

  useEffect(() => {
    if (!transcript) {
      return;
    }

    setInputText(transcript);
    setDetectedEmotion(detectEmotion(transcript));
    setSelectedTone((currentTone) => inferTone(transcript, currentTone));

    if (meetingMode) {
      setMeetingTranscript((previous) => {
        if (previous.length && previous[previous.length - 1].text === transcript) {
          return previous;
        }

        return [
          ...previous,
          {
            id: Date.now(),
            text: transcript,
            timestamp: new Date().toLocaleTimeString(),
          },
        ].slice(-12);
      });
    }
  }, [meetingMode, transcript]);

  useEffect(() => {
    let interval;

    if (isDetectingSign) {
      interval = setInterval(async () => {
        try {
          const response = await axios.get('http://localhost:8000/detection_status');
          if (response.data.sentence_buffer) {
            setInputText(response.data.sentence_buffer);
            setDetectedEmotion(detectEmotion(response.data.sentence_buffer));
          }
          if (!response.data.active) {
            setIsDetectingSign(false);
          }
        } catch (error) {
          console.error('Detection poll error:', error);
        }
      }, 1000);
    }

    return () => clearInterval(interval);
  }, [isDetectingSign]);

  const updateDraft = (value) => {
    setInputText(value);
    setDetectedEmotion(detectEmotion(value));
    setConversationSignals((currentSignals) => buildConversationSignals({
      text: value,
      messages,
      telemetry: typingSessionRef.current,
      assertivenessLevel: currentSignals.assertiveness_level,
    }));
  };

  const trackTypingTelemetry = (event) => {
    const now = Date.now();
    if (typingSessionRef.current.lastTimestamp) {
      const interval = now - typingSessionRef.current.lastTimestamp;
      typingSessionRef.current.keyIntervals = [...typingSessionRef.current.keyIntervals.slice(-24), interval];
    }
    typingSessionRef.current.lastTimestamp = now;

    if (event.key === 'Backspace') {
      typingSessionRef.current.backspaceCount += 1;
    } else if (event.key.length === 1 || event.key === ' ') {
      typingSessionRef.current.charactersTyped += 1;
    }
  };

  const resetTypingTelemetry = () => {
    typingSessionRef.current = {
      lastTimestamp: null,
      keyIntervals: [],
      backspaceCount: 0,
      charactersTyped: 0,
    };
  };

  const toggleSignDetection = async () => {
    setChatError('');

    if (isDetectingSign) {
      await axios.post('http://localhost:8000/stop_detection');
      setIsDetectingSign(false);
      return;
    }

    try {
      await axios.post('http://localhost:8000/start_detection', { language: detectionLang });
      setIsDetectingSign(true);
    } catch (error) {
      setChatError('Sign detection service is not reachable on localhost:8000.');
    }
  };

  const processWithLLM = async ({ text, sourceLanguage, targetLanguage, tone, emotion, inputMode, adaptiveSignals }) => {
    setIsTranslating(true);
    setChatError('');

    const sourceMeta = getLanguageMeta(sourceLanguage);
    const targetMeta = getLanguageMeta(targetLanguage);
    const translationInstruction = [
      `Translate the user's message from ${sourceMeta.name} to ${targetMeta.name}.`,
      `Preserve the meaning, but rewrite it in a ${tone} tone.`,
      `Return the final answer only in ${targetMeta.name}.`,
      'Do not mix in English unless the source explicitly contains names or technical terms.',
      adaptiveSignals.cognitive_load === 'high'
        ? 'The receiver appears cognitively overloaded, so simplify vocabulary and use shorter sentences.'
        : adaptiveSignals.cognitive_load === 'medium'
          ? 'Prefer moderate simplicity and reduce unnecessary complexity.'
          : 'Keep the translation natural and complete.',
      adaptiveSignals.sarcasm_detected
        ? 'Sarcasm or irony may be present. Preserve intent carefully and avoid literal mistranslation.'
        : 'No strong sarcasm signal detected.',
      adaptiveSignals.code_switching_detected
        ? 'The input may contain code-switching across languages or scripts. Preserve mixed-language intent carefully.'
        : 'Treat the source as primarily one language.',
      adaptiveSignals.relationship_state === 'friend'
        ? 'Conversation history suggests friends, so relaxed familiarity is acceptable.'
        : adaptiveSignals.relationship_state === 'colleague'
          ? 'Conversation history suggests colleagues, so keep collaborative professionalism.'
          : 'Conversation history suggests unfamiliar participants, so remain politely neutral.',
      adaptiveSignals.assertiveness_level > 0
        ? `Strengthen confidence slightly with assertiveness level ${adaptiveSignals.assertiveness_level}.`
        : adaptiveSignals.assertiveness_level < 0
          ? `Soften the wording slightly with assertiveness level ${adaptiveSignals.assertiveness_level}.`
          : 'Keep assertiveness balanced.',
    ].join(' ');

    try {
      const response = await axios.post('http://localhost:8000/process_chat', {
        text,
        source_lang: sourceMeta.name,
        source_lang_code: sourceLanguage,
        source_lang_short: sourceMeta.shortCode,
        target_lang: targetMeta.name,
        target_lang_code: targetLanguage,
        target_lang_short: targetMeta.shortCode,
        tone,
        emotion,
        input_mode: inputMode,
        instruction: translationInstruction,
        cognitive_load: adaptiveSignals.cognitive_load,
        sarcasm_detected: adaptiveSignals.sarcasm_detected,
        relationship_state: adaptiveSignals.relationship_state,
        code_switching_detected: adaptiveSignals.code_switching_detected,
        paralinguistic_signals: adaptiveSignals.paralinguistic_signals,
        assertiveness_level: adaptiveSignals.assertiveness_level,
        research_signals: adaptiveSignals,
      });

      const parsedResponse = extractProcessedResponse(response.data, text);
      if (parsedResponse.method === 'advanced_simulation' || parsedResponse.method === 'unavailable') {
        throw new Error(
          parsedResponse.message ||
          'The Python translation API is running without a real LLM configuration, so it cannot produce actual target-language translation.'
        );
      }

      return {
        ...parsedResponse,
        translationVerified: looksTranslated(parsedResponse.translatedText, targetMeta),
      };
    } catch (error) {
      console.error('LLM processing error:', error);
      const backendMessage = error?.response?.data?.message || error?.message;
      setChatError(backendMessage || 'Translation service is unavailable. Showing your original message for now.');
      return {
        translatedText: text,
        detectedTone: tone,
        detectedEmotion: emotion,
        translationVerified: false,
        appliedControls: adaptiveSignals,
      };
    } finally {
      setIsTranslating(false);
    }
  };

  const startMeetingCapture = async () => {
    setMeetingMode(true);
    setMeetingStatus('Listening');
    setChatError('');

    if (browserSupportsSpeechRecognition && !listening) {
      SpeechRecognition.startListening({ continuous: true, language: sourceLang });
    }

    setIsCameraEnabled(true);
  };

  const stopMeetingCapture = () => {
    setMeetingStatus('Stopped');
    setMeetingMode(false);
    setIsCameraEnabled(false);
    SpeechRecognition.stopListening();
  };

  const handleSendMessage = async () => {
    const currentInput = inputText.trim();
    if (!currentInput) {
      return;
    }

    const messageTone = inferTone(currentInput, selectedTone);
    const messageEmotion = detectEmotion(currentInput);
    const inputMode = listening ? 'voice' : isDetectingSign ? 'sign' : 'text';
    const adaptiveSignals = buildConversationSignals({
      text: currentInput,
      messages,
      telemetry: typingSessionRef.current,
      assertivenessLevel,
    });

    if (listening) {
      SpeechRecognition.stopListening();
    }

    const userMessage = {
      id: Date.now(),
      text: currentInput,
      sender: 'user',
      emotion: messageEmotion,
      originalLang: sourceLang,
      targetLang,
      tone: messageTone,
      inputMode,
      adaptiveSignals,
    };

    setMessages((previous) => [...previous, userMessage]);
    setInputText('');
    setDetectedEmotion('Neutral 😐');
    setConversationSignals(adaptiveSignals);
    resetTranscript();
    resetTypingTelemetry();

    const llmResult = await processWithLLM({
      text: currentInput,
      sourceLanguage: sourceLang,
      targetLanguage: targetLang,
      tone: messageTone,
      emotion: messageEmotion,
      inputMode,
      adaptiveSignals,
    });

    const botReply = {
      id: Date.now() + 1,
      text: currentInput,
      translatedText: llmResult.translatedText,
      sender: 'bot',
      emotion: llmResult.detectedEmotion || messageEmotion,
      originalLang: sourceLang,
      targetLang,
      tone: llmResult.detectedTone || messageTone,
      inputMode,
      translationVerified: llmResult.translationVerified,
      adaptiveSignals,
      appliedControls: llmResult.appliedControls || adaptiveSignals,
    };

    setMessages((previous) => [...previous, botReply]);

    if (!llmResult.translationVerified) {
      const targetMeta = getLanguageMeta(targetLang);
      setChatError(`The service responded, but the output does not look fully translated to ${targetMeta.name}. The backend may still be ignoring the target language.`);
    }
  };

  const getMatchingVoice = (lang) =>
    availableVoices.find((voice) => {
      const voiceLang = voice.lang?.toLowerCase() || '';
      const targetLang = lang.toLowerCase();
      const shortTargetLang = targetLang.split('-')[0];

      return voiceLang === targetLang || voiceLang.startsWith(`${shortTargetLang}-`) || voiceLang === shortTargetLang;
    });

  const speakWithBackend = async (text, languageMeta) => {
    try {
      setChatError('');
      const response = await axios.post(
        'http://localhost:8000/speak_text',
        {
          text,
          language: languageMeta.name,
          language_code: languageMeta.shortCode,
        },
        { responseType: 'blob' }
      );

      if (audioRef.current) {
        audioRef.current.pause();
      }
      if (audioUrlRef.current) {
        URL.revokeObjectURL(audioUrlRef.current);
      }

      const audioUrl = URL.createObjectURL(response.data);
      audioUrlRef.current = audioUrl;
      const audio = new Audio(audioUrl);
      audioRef.current = audio;
      await audio.play();
    } catch (error) {
      const backendMessage = `Speaker audio is unavailable for ${languageMeta.name} right now. Check that the Python API is running and gTTS is installed.`;
      setChatError(backendMessage);
    }
  };

  const speak = async (text, lang, tone) => {
    const languageMeta = getLanguageMeta(lang);
    const shortLanguage = languageMeta.shortCode;
    const matchingVoice = getMatchingVoice(lang);

    if (BACKEND_TTS_LANGUAGES.has(shortLanguage)) {
      await speakWithBackend(text, languageMeta);
      return;
    }

    if (!matchingVoice) {
      setChatError(`Speaker is unavailable for ${languageMeta.name} on this browser. Install a ${languageMeta.name} text-to-speech voice in Windows/browser settings.`);
      return;
    }

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = lang;
    utterance.voice = matchingVoice;

    if (tone === 'Empathetic') {
      utterance.rate = 0.92;
      utterance.pitch = 0.95;
    } else if (tone === 'Casual') {
      utterance.rate = 1.02;
      utterance.pitch = 1.1;
    } else if (tone === 'Professional') {
      utterance.rate = 0.96;
      utterance.pitch = 1;
    }

    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utterance);
  };

  const hasVoiceForLanguage = (lang) => {
    const languageMeta = getLanguageMeta(lang);
    if (BACKEND_TTS_LANGUAGES.has(languageMeta.shortCode)) {
      return true;
    }

    return Boolean(getMatchingVoice(lang));
  };

  const sourceMeta = getLanguageMeta(sourceLang);
  const targetMeta = getLanguageMeta(targetLang);

  return (
    <div className="chat-container">
      <div className={`settings-panel ${showSettings ? 'show' : ''}`}>
        <div className="settings-header">
          <h3><Settings size={20} /> Translation Settings</h3>
          <button type="button" className="panel-close-btn" onClick={() => setShowSettings(false)}>×</button>
        </div>

        <div className="setting-group">
          <label htmlFor="source-language">Source Language</label>
          <select id="source-language" value={sourceLang} onChange={(event) => setSourceLang(event.target.value)}>
            {languages.map((language) => (
              <option key={language.code} value={language.code}>
                {language.flag} {language.name}
              </option>
            ))}
          </select>
        </div>

        <div className="setting-group">
          <label htmlFor="target-language">Target Language</label>
          <select id="target-language" value={targetLang} onChange={(event) => setTargetLang(event.target.value)}>
            {languages.map((language) => (
              <option key={language.code} value={language.code}>
                {language.flag} {language.name}
              </option>
            ))}
          </select>
        </div>

        <div className="setting-group">
          <label htmlFor="sign-language">Sign Language Input</label>
          <select id="sign-language" value={detectionLang} onChange={(event) => setDetectionLang(event.target.value)}>
            <option value="ASL">ASL (American)</option>
            <option value="ISL">ISL (Indian)</option>
            <option value="TSL">TSL (Tamil)</option>
          </select>
        </div>

        <div className="setting-group">
          <label>Translation Tone</label>
          <div className="tone-grid">
            {tones.map((tone) => (
              <button
                key={tone.label}
                type="button"
                className={`tone-card ${selectedTone === tone.label ? 'active' : ''}`}
                onClick={() => setSelectedTone(tone.label)}
                title={tone.description}
              >
                <span className="tone-icon">{tone.icon}</span>
                <span className="tone-label">{tone.label}</span>
              </button>
            ))}
          </div>
        </div>

        <div className="setting-group">
          <label htmlFor="assertiveness-slider">Negotiation / Assertiveness</label>
          <input
            id="assertiveness-slider"
            type="range"
            min="-2"
            max="2"
            step="1"
            value={assertivenessLevel}
            onChange={(event) => {
              const value = Number(event.target.value);
              setAssertivenessLevel(value);
              setConversationSignals((currentSignals) => ({
                ...currentSignals,
                assertiveness_level: value,
              }));
            }}
            className="assertiveness-slider"
          />
          <div className="assertiveness-row">
            <span>Soften</span>
            <strong>{getAssertivenessLabel(assertivenessLevel)}</strong>
            <span>Assert</span>
          </div>
        </div>
      </div>

      <div className="chat-main">
        <header className="chat-header">
          <div className="header-info">
            <div className="avatar-main">
              <Sparkles className="sparkle-icon" size={24} />
            </div>
            <div>
              <h2>Multimodal LLM Live Chat</h2>
              <p>Real-time translation, better tone handling, and cleaner conversation flow</p>
            </div>
          </div>
          <button type="button" className="settings-toggle" onClick={() => setShowSettings((open) => !open)}>
            <Settings size={24} />
          </button>
        </header>

        <div className="messages-list">
          <div className="meeting-panel">
            <div className="meeting-header">
              <div>
                <h3>Meeting Capture</h3>
                <p>Capture spoken discussion with live transcript and optional camera preview.</p>
              </div>
              <div className="meeting-actions">
                {!meetingMode ? (
                  <button type="button" className="meeting-btn primary" onClick={startMeetingCapture}>
                    <CircleDot size={16} /> Start Meeting
                  </button>
                ) : (
                  <button type="button" className="meeting-btn danger" onClick={stopMeetingCapture}>
                    <Square size={16} /> Stop Meeting
                  </button>
                )}
                <button
                  type="button"
                  className={`meeting-btn ${isCameraEnabled ? 'active' : ''}`}
                  onClick={() => setIsCameraEnabled((current) => !current)}
                >
                  <Camera size={16} /> Camera
                </button>
              </div>
            </div>

            <div className="meeting-body">
              <div className="meeting-preview">
                {isCameraEnabled ? (
                  <Webcam
                    ref={webcamRef}
                    audio={false}
                    screenshotFormat="image/jpeg"
                    videoConstraints={{ facingMode: 'user' }}
                    className="meeting-webcam"
                  />
                ) : (
                  <div className="meeting-placeholder">
                    <Camera size={22} />
                    <span>Camera preview is off</span>
                  </div>
                )}
                <div className="meeting-status">
                  <span className={`status-dot ${meetingMode ? 'live' : ''}`}></span>
                  {meetingStatus}
                </div>
              </div>

              <div className="meeting-transcript">
                <div className="meeting-transcript-header">
                  <span>Live Meeting Notes</span>
                  <span>{meetingTranscript.length} lines</span>
                </div>
                <div className="meeting-transcript-list">
                  {meetingTranscript.length ? (
                    meetingTranscript.map((entry) => (
                      <div key={entry.id} className="meeting-line">
                        <span>{entry.timestamp}</span>
                        <p>{entry.text}</p>
                      </div>
                    ))
                  ) : (
                    <div className="meeting-empty">Start meeting mode to capture speech continuously.</div>
                  )}
                </div>
              </div>
            </div>
          </div>

          <div className="insight-panel">
            <div className="insight-header">
              <h3>Adaptive Intelligence Signals</h3>
              <p>Research-safe hooks for cognitive load, sarcasm, persona drift, code-switching, and business tone equalization.</p>
            </div>
            <div className="insight-badges">
              <span className="insight-badge">Load: {conversationSignals.cognitive_load}</span>
              <span className="insight-badge">Relationship: {conversationSignals.relationship_state}</span>
              <span className={`insight-badge ${conversationSignals.sarcasm_detected ? 'warn' : ''}`}>
                Sarcasm: {conversationSignals.sarcasm_detected ? 'possible' : 'low'}
              </span>
              <span className={`insight-badge ${conversationSignals.code_switching_detected ? 'active' : ''}`}>
                Code-switch: {conversationSignals.code_switching_detected ? 'detected' : 'single language'}
              </span>
              <span className="insight-badge">Assertiveness: {getAssertivenessLabel(assertivenessLevel)}</span>
              <span className="insight-badge">Emoji cues: {conversationSignals.paralinguistic_signals.emoji_count}</span>
            </div>
          </div>

          {messages.map((message) => (
            <div key={message.id} className={`message-wrapper ${message.sender}`}>
              <div className="message-bubble">
                <div className="message-meta">
                  <span>{message.sender === 'user' ? 'You' : 'AI Translation'}</span>
                  <span>{message.inputMode}</span>
                </div>

                {message.sender === 'user' ? (
                  <div className="original-text">{message.text}</div>
                ) : (
                  <div className="translated-block">
                    <div className="source-text">
                      <div className="translation-badge">
                        <Languages size={12} /> Source ({message.originalLang.split('-')[0]})
                      </div>
                      <p>{message.text}</p>
                    </div>
                    <div className="translated-text">
                      <div className="translation-badge">
                        <Languages size={12} /> Translated ({message.targetLang.split('-')[0]})
                      </div>
                      <p>{message.translatedText}</p>
                    </div>
                  </div>
                )}

                <div className="message-footer">
                  <span className="emotion-tag">{message.emotion}</span>
                  <span className="tone-tag">{message.tone}</span>
                  {message.sender === 'bot' && (
                    <span className={`translation-state ${message.translationVerified ? 'ok' : 'warn'}`}>
                      {message.translationVerified ? 'Target language detected' : 'Target language unclear'}
                    </span>
                  )}
                  <button
                    type="button"
                    className="speak-btn"
                    disabled={!hasVoiceForLanguage(message.sender === 'user' ? message.originalLang : message.targetLang)}
                    title={
                      hasVoiceForLanguage(message.sender === 'user' ? message.originalLang : message.targetLang)
                        ? (
                          BACKEND_TTS_LANGUAGES.has(getLanguageMeta(message.sender === 'user' ? message.originalLang : message.targetLang).shortCode)
                            ? 'Listen using backend audio'
                            : 'Listen to this message'
                        )
                        : `No installed voice for ${getLanguageMeta(message.sender === 'user' ? message.originalLang : message.targetLang).name}`
                    }
                    onClick={() =>
                      speak(
                        message.sender === 'user' ? message.text : message.translatedText,
                        message.sender === 'user' ? message.originalLang : message.targetLang,
                        message.tone
                      )
                    }
                  >
                    <Volume2 size={14} />
                  </button>
                </div>

                {message.sender === 'bot' && message.appliedControls && (
                  <div className="applied-controls">
                    <span className="control-chip">Tone: {message.appliedControls.tone || message.tone}</span>
                    <span className="control-chip">Load: {message.appliedControls.cognitive_load || 'low'}</span>
                    <span className="control-chip">Assert: {getAssertivenessLabel(message.appliedControls.assertiveness_level ?? 0)}</span>
                    <span className="control-chip">Relation: {message.appliedControls.relationship_state || 'stranger'}</span>
                  </div>
                )}
              </div>
            </div>
          ))}

          {isTranslating && (
            <div className="message-wrapper bot">
              <div className="message-bubble typing">
                <div className="dot"></div>
                <div className="dot"></div>
                <div className="dot"></div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        <footer className="chat-input-area">
          <div className="input-toolbar">
            <div className="active-settings">
              <span>{sourceMeta.flag} {sourceMeta.name}</span>
              <span className="sep">-></span>
              <span>{targetMeta.flag} {targetMeta.name}</span>
              <span className="sep">|</span>
              <span>Tone: {selectedTone}</span>
              <span className="sep">|</span>
              <span>Emotion: {detectedEmotion}</span>
            </div>
          </div>

          {!browserSupportsSpeechRecognition && (
            <div className="status-banner warning">
              <AlertCircle size={16} />
              <span>This browser does not support speech recognition.</span>
            </div>
          )}

          {chatError && (
            <div className="status-banner error">
              <AlertCircle size={16} />
              <span>{chatError}</span>
            </div>
          )}

          <div className="input-field-wrapper">
            <button
              type="button"
              className={`sign-btn ${isDetectingSign ? 'active' : ''}`}
              onClick={toggleSignDetection}
              title="Start Sign Language Detection"
            >
              {isDetectingSign ? <VideoOff size={22} /> : <Video size={22} />}
            </button>

            <button
              type="button"
              className={`mic-btn ${listening ? 'listening' : ''}`}
              onClick={() =>
                listening
                  ? SpeechRecognition.stopListening()
                  : SpeechRecognition.startListening({ continuous: true, language: sourceLang })
              }
              disabled={!browserSupportsSpeechRecognition}
              title={listening ? 'Stop voice input' : 'Start voice input'}
            >
              {listening ? <MicOff size={22} /> : <Mic size={22} />}
            </button>

            <input
              type="text"
              placeholder="Type or speak your message here..."
              value={inputText}
              onChange={(event) => updateDraft(event.target.value)}
              onKeyDown={(event) => {
                trackTypingTelemetry(event);
                if (event.key === 'Enter') {
                  handleSendMessage();
                }
              }}
            />

            <button type="button" className="send-btn" onClick={handleSendMessage} disabled={!inputText.trim() || isTranslating}>
              <Send size={22} />
            </button>
          </div>

          <p className="support-hint">
            Multimodal chat now keeps source and translated text together, uses better tone hints, and sends richer metadata for text, voice, and sign input.
          </p>
        </footer>
      </div>
    </div>
  );
};

export default ChatTranslation;
