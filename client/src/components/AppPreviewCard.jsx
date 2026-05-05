import React, { useState } from 'react';

export default function AppPreviewCard({ data, onSendMessage }) {
  const [isEditing, setIsEditing] = useState(false);
  const [editInstruction, setEditInstruction] = useState('');

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

  return (
    <div className="w-full glass-panel border border-rent-border rounded-xl overflow-hidden mt-2 shadow-soft font-sans">
      
      
      <div className="bg-rent-elevated p-4 border-b border-rent-border flex justify-between items-start">
        <div>
          <h3 className="text-white font-bold text-lg">{data.appName || 'Untitled App'}</h3>
          <p className="text-white/50 text-sm mt-1">{data.appDescription}</p>
        </div>
        <div className="bg-rent-purple/10 border border-rent-purple/30 px-3 py-1 rounded-full flex items-center gap-1 shrink-0 ml-2">
          <span className="text-rent-purple text-sm font-medium">{data.cost} coins / run</span>
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
              className="btn-cta px-5 py-2.5 text-white rounded-lg text-sm font-bold transition-transform hover:scale-[1.02] active:scale-95 shadow-lg shadow-rent-purple/20"
            >
              🚀 Publish to Marketplace
            </button>
            <button 
              onClick={() => onSendMessage('Save Draft')}
              className="px-4 py-2.5 bg-transparent text-white/50 hover:text-white rounded-lg text-sm font-medium transition-colors"
            >
              Save Draft
            </button>
            <button 
              onClick={() => setIsEditing(true)}
              className="px-4 py-2 bg-rent-border/50 text-white/70 hover:bg-rent-border hover:text-white rounded-lg text-sm font-medium ml-auto transition-colors"
            >
              ✏️ Tweak Setup
            </button>
          </div>
        ) : (
          <div className="flex gap-2">
            <input 
              type="text" 
              value={editInstruction}
              onChange={(e) => setEditInstruction(e.target.value)}
              placeholder="Tell the AI what to change..."
              className="flex-1 bg-black/50 border border-rent-border rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-rent-purple transition-colors"
              onKeyDown={(e) => e.key === 'Enter' && handleEditSubmit()}
            />
            <button 
              onClick={handleEditSubmit}
              className="px-4 py-2 bg-rent-purple text-white rounded-lg text-sm font-medium hover:bg-[#8b6ffe] transition-colors"
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
      </div>
    </div>
  );
}
