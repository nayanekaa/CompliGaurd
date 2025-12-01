import React, { useState, useRef, useEffect } from 'react';
import { Send, User, Bot, Paperclip, FileText, AlertTriangle, ShieldCheck, CheckCircle2, AlertCircle, HelpCircle } from 'lucide-react';
import { Message, Citation } from '../types';
import { generateComplianceResponse } from '../services/geminiService';

export const ComplianceChat: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '0',
      role: 'system',
      content: '',
      timestamp: new Date(),
      structuredResponse: {
        answer: "Hello! I'm CompliGuard, your auditable compliance assistant. I can answer questions based on internal policies and ISO/GDPR regulations. All my answers are backed by citations.",
        confidence: 'High',
        citations: [],
        suggestedAction: ''
      }
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await generateComplianceResponse(userMessage.content);
      
      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response.text, // raw fallback
        structuredResponse: response.structuredResponse,
        timestamp: new Date(),
      };
      
      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error(error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: "I encountered an error connecting to the policy engine. Please check your API key or try again.",
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full bg-slate-50 relative">
      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-6 space-y-8 pb-32">
        {messages.map((msg) => (
          <div key={msg.id} className={`flex gap-4 ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
            
            {/* Avatar */}
            <div className={`w-10 h-10 rounded-full flex items-center justify-center shrink-0 border 
              ${msg.role === 'user' ? 'bg-white border-slate-200' : 'bg-blue-600 border-blue-700'}`}>
              {msg.role === 'user' ? <User className="w-6 h-6 text-slate-600" /> : <Bot className="w-6 h-6 text-white" />}
            </div>

            {/* Content Bubble */}
            <div className={`max-w-3xl rounded-2xl shadow-sm border overflow-hidden
              ${msg.role === 'user' ? 'bg-blue-50 border-blue-100 text-slate-800' : 'bg-white border-slate-200'}`}>
              
              {/* Main Text */}
              <div className="p-5 text-sm md:text-base leading-relaxed whitespace-pre-wrap">
                {msg.structuredResponse ? msg.structuredResponse.answer : msg.content}
              </div>

              {/* Structured Elements for Assistant */}
              {msg.role === 'assistant' && msg.structuredResponse && (
                <div className="bg-slate-50 border-t border-slate-100 p-4 space-y-4">
                  
                  {/* Evidence Section */}
                  {msg.structuredResponse.citations.length > 0 && (
                    <div>
                      <h4 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-2 flex items-center gap-1">
                        <FileText className="w-3 h-3" /> Evidence & Sources
                      </h4>
                      <div className="space-y-2">
                        {msg.structuredResponse.citations.map((cite, idx) => (
                          <div key={idx} className="bg-white border border-slate-200 rounded-lg p-3 hover:border-blue-300 transition-colors cursor-pointer group">
                            <div className="flex justify-between items-start mb-1">
                                <span className="text-xs font-semibold text-blue-700 font-mono bg-blue-50 px-1.5 py-0.5 rounded">{cite.source}</span>
                                {cite.page && <span className="text-xs text-slate-400">Page {cite.page}</span>}
                            </div>
                            <p className="text-xs text-slate-600 italic">"{cite.excerpt}"</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Footer Stats: Confidence & Action */}
                  <div className="flex flex-wrap items-center gap-4 pt-2">
                    
                    {/* Confidence Badge */}
                    <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium border
                      ${msg.structuredResponse.confidence === 'High' ? 'bg-emerald-50 text-emerald-700 border-emerald-200' : 
                        msg.structuredResponse.confidence === 'Medium' ? 'bg-amber-50 text-amber-700 border-amber-200' : 
                        'bg-red-50 text-red-700 border-red-200'}`}>
                      {msg.structuredResponse.confidence === 'High' ? <ShieldCheck className="w-3.5 h-3.5" /> : <AlertTriangle className="w-3.5 h-3.5" />}
                      Confidence: {msg.structuredResponse.confidence}
                    </div>

                    {/* Suggested Action */}
                    {msg.structuredResponse.suggestedAction && (
                       <div className="flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium bg-indigo-50 text-indigo-700 border border-indigo-200">
                          <CheckCircle2 className="w-3.5 h-3.5" />
                          Suggested: {msg.structuredResponse.suggestedAction}
                       </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="flex gap-4">
             <div className="w-10 h-10 bg-blue-600 rounded-full flex items-center justify-center shrink-0">
               <Bot className="w-6 h-6 text-white" />
             </div>
             <div className="bg-white border border-slate-200 rounded-2xl p-4 shadow-sm flex items-center gap-2">
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                <span className="text-sm text-slate-500 ml-2">Searching knowledge base...</span>
             </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="absolute bottom-0 left-0 right-0 bg-white border-t border-slate-200 p-6 z-10">
        <div className="max-w-4xl mx-auto relative shadow-xl rounded-xl">
            <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        handleSend();
                    }
                }}
                placeholder="Ask a compliance question (e.g. 'What is the sick leave policy for >7 days?')"
                className="w-full bg-white text-slate-900 placeholder-slate-400 rounded-xl pl-4 pr-14 py-4 focus:outline-none focus:ring-2 focus:ring-blue-500/50 resize-none border border-slate-200"
                rows={1}
                style={{ minHeight: '60px' }}
            />
            <div className="absolute right-2 bottom-2.5 flex items-center gap-1">
                <button className="p-2 text-slate-400 hover:text-blue-600 transition-colors rounded-lg hover:bg-slate-50">
                    <Paperclip className="w-5 h-5" />
                </button>
                <button 
                    onClick={handleSend}
                    disabled={!input.trim() || isLoading}
                    className="p-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed shadow-md"
                >
                    <Send className="w-5 h-5" />
                </button>
            </div>
        </div>
        <div className="max-w-4xl mx-auto mt-2 text-center">
            <p className="text-xs text-slate-400 flex items-center justify-center gap-1">
                <ShieldCheck className="w-3 h-3" /> All answers are auditable and generated from internal policies v2.4.0
            </p>
        </div>
      </div>
    </div>
  );
};
