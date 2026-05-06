import React, { useState } from 'react';
import { testPromptRun } from '../utils/api';

export default function AppPreviewCard({ data, onSendMessage }) {
  const [isEditing, setIsEditing] = useState(false);
  const [editInstruction, setEditInstruction] = useState('');
  const [isPreviewMode, setIsPreviewMode] = useState(false);
  const [testInputs, setTestInputs] = useState({});
  const [testOutput, setTestOutput] = useState("");
  const [testError, setTestError] = useState("");
  const [isRunningTest, setIsRunningTest] = useState(false);

  if (!data) return null;

  // Format variables beautifully
  const formatPrompt = (text) => {
    if (!text) return null;
    const parts = text.split(/(\$\$\w+)/g);
    return parts.map((part, i) => 
      part.startsWith('$$') ? <span key={i} className="text-rent-purple font-semibold">{part}</span> : part
    );
  };

  const handleEditSubmit = () => {
    if (editInstruction.trim()) {
      onSendMessage(`Change: ${editInstruction}`);
      setIsEditing(false);
      setEditInstruction('');
    }
  };

  // Ensure variables array exists
  const variables = data.variablesUsed || [];
  const normalizedVariables = variables.map((v) => String(v || "").replace(/^\$\$/, "")).filter(Boolean);

  const handleInputChange = (key, value) => {
    setTestInputs((prev) => ({ ...prev, [key]: value }));
  };

  const handleRunTest = async () => {
    setIsRunningTest(true);
    setTestError("");
    try {
      const result = await testPromptRun({
        systemPrompt: data.systemPrompt,
        userPrompt: data.userPrompt,
        testInputs,
      });
      setTestOutput(result.output || "");
    } catch (error) {
      setTestError(error.message || "Test run failed.");
    } finally {
      setIsRunningTest(false);
    }
  };

  return (
    <div className="w-full glass-panel border border-rent-border rounded-xl overflow-hidden mt-2 shadow-soft font-sans">
      
      
      <div className="bg-rent-elevated p-4 border-b border-rent-border flex justify-between items-start">
        <div>
          <h3 className="text-white font-bold text-lg">{data.appName || 'Untitled App'}</h3>
          <p className="text-white/50 text-sm mt-1">{data.appDescription}</p>
        </div>
        <div className="bg-[#2d1b4e] border border-[#5a32a3] px-3 py-1 rounded-full flex items-center gap-1 shrink-0 ml-2">
          <span className="text-[#a77bf3] text-sm font-medium">{data.cost} coins / run</span>
        </div>
      </div>

      
      <div className="p-4 space-y-4">
        <div>
          <span className="text-xs font-semibold text-white/40 uppercase tracking-wider block mb-2">Backend Logic</span>
          <div className="bg-black/50 border border-rent-border p-3 rounded-lg text-sm text-white/70 font-mono leading-relaxed h-24 overflow-y-auto custom-scrollbar">
            {data.systemPrompt}
          </div>
        </div>
        <div>
          <span className="text-xs font-semibold text-white/40 uppercase tracking-wider block mb-2">Prompt Template</span>
          <div className="bg-black/50 border border-rent-border p-3 rounded-lg text-sm text-white/70 font-mono leading-relaxed custom-scrollbar">
            {formatPrompt(data.userPrompt)}
          </div>
        </div>
      </div>

      
      <div className="p-4 border-t border-rent-border bg-rent-elevated">
        {!isEditing ? (
          <div className="flex flex-wrap gap-2 items-center">
            <button 
              onClick={() => onSendMessage('Publish App')}
              className="px-5 py-2.5 bg-[#6d28d9] hover:bg-[#5b21b6] text-white rounded-lg text-sm font-medium transition-colors"
            >
              Publish to Marketplace
            </button>
            <button 
              onClick={() => onSendMessage('Save Draft')}
              className="px-4 py-2.5 bg-transparent text-gray-400 hover:text-white rounded-lg text-sm font-medium transition-colors"
            >
              Save Draft
            </button>
            <button 
              onClick={() => setIsEditing(true)}
              className="px-4 py-2 bg-[#2a2a2a] text-gray-300 hover:text-white rounded-lg text-sm font-medium ml-auto transition-colors"
            >
              Edit
            </button>
            <button
              onClick={() => setIsPreviewMode((prev) => !prev)}
              className="px-4 py-2 bg-[#2a2a2a] text-gray-300 hover:text-white rounded-lg text-sm font-medium transition-colors"
            >
              {isPreviewMode ? "Hide Test" : "Test App"}
            </button>
          </div>
        ) : (
          <div className="flex gap-2">
            <input 
              type="text" 
              value={editInstruction}
              onChange={(e) => setEditInstruction(e.target.value)}
              placeholder="Tell the AI what to change..."
              className="flex-1 bg-black/50 border border-rent-border rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-[#8b5cf6] transition-colors"
              onKeyDown={(e) => e.key === 'Enter' && handleEditSubmit()}
            />
            <button 
              onClick={handleEditSubmit}
              className="px-4 py-2 bg-[#6d28d9] text-white rounded-lg text-sm font-medium hover:bg-[#5b21b6] transition-colors"
            >
              Update
            </button>
            <button 
              onClick={() => setIsEditing(false)}
              className="px-3 py-2 text-white/50 hover:text-white text-sm transition-colors"
            >
              Cancel
            </button>
          </div>
        )}

        {isPreviewMode ? (
          <div className="mt-4 border-t border-rent-border pt-4 space-y-3">
            <div className="text-sm text-white/80 font-medium">Live Preview / Test Run</div>
            <div className="space-y-2">
              {normalizedVariables.map((name) => (
                <input
                  key={name}
                  type="text"
                  value={testInputs[name] || ""}
                  onChange={(e) => handleInputChange(name, e.target.value)}
                  placeholder={`Enter ${name}`}
                  className="w-full bg-black/50 border border-rent-border rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-[#8b5cf6]"
                />
              ))}
            </div>
            <button
              onClick={handleRunTest}
              disabled={isRunningTest}
              className="px-4 py-2 bg-[#6d28d9] text-white rounded-lg text-sm font-medium hover:bg-[#5b21b6] disabled:opacity-60"
            >
              {isRunningTest ? "Running..." : "Run Test"}
            </button>
            {testError ? <div className="text-red-300 text-xs">{testError}</div> : null}
            {testOutput ? (
              <div className="bg-black/50 border border-rent-border p-3 rounded-lg text-sm text-white/80 whitespace-pre-wrap">
                {testOutput}
              </div>
            ) : null}
          </div>
        ) : null}
      </div>
    </div>
  );
}
