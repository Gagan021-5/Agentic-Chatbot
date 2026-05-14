import React, { useEffect, useMemo, useState } from "react";

export default function MultiSelectFormCard({ data, onSendMessage, isLoading }) {
  const options = useMemo(() => (Array.isArray(data?.options) ? data.options : []), [data]);
  const initialVariables = useMemo(
    () =>
      (Array.isArray(data?.variables) ? data.variables : [])
        .map((variable) => {
          if (typeof variable === "string") {
            return { name: variable, placeholder: "Enter details...", test_value: "" };
          }
          return {
            name: String(variable?.name || "").trim(),
            placeholder: String(variable?.placeholder || "Enter details...").trim(),
            test_value: String(variable?.test_value || "").trim()
          };
        })
        .filter((variable) => variable.name),
    [data]
  );
  const [selectedOptions, setSelectedOptions] = useState(options.slice(0, 2));
  const variables = initialVariables;
  const [variableValues, setVariableValues] = useState(() => {
    const initialVals = {};
    initialVariables.forEach((v) => {
      if (v.test_value) {
        initialVals[v.name] = v.test_value;
      }
    });
    return initialVals;
  });

  useEffect(() => {
    setVariableValues((prev) => {
      let changed = false;
      const next = { ...prev };
      initialVariables.forEach((v) => {
        if (v.test_value && !(v.name in next)) {
          next[v.name] = v.test_value;
          changed = true;
        }
      });
      return changed ? next : prev;
    });
  }, [initialVariables]);

  const toggleOption = (option) => {
    setSelectedOptions((prev) =>
      prev.includes(option) ? prev.filter((item) => item !== option) : [...prev, option]
    );
  };

  const updateVariableValue = (name, value) => {
    setVariableValues((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = () => {
    const payload = {
      selectedOptions: selectedOptions.filter(Boolean),
      variables: variables.map((variable) => ({
        name: variable.name,
        placeholder: variable.placeholder || "Enter details...",
        value: String(variableValues[variable.name] || "").trim()
      }))
    };
    onSendMessage(`multi_select_form::${JSON.stringify(payload)}`);
  };

  return (
    <div className="w-full rounded-xl border border-rent-border bg-rent-elevated/40 p-4 space-y-4">
      <div>
        <h4 className="text-white font-semibold text-sm">Relevant Features</h4>
        <p className="text-white/50 text-xs mt-1">Select what should be included in your app.</p>
      </div>

      <div className="flex flex-wrap gap-2">
        {options.map((option) => {
          const active = selectedOptions.includes(option);
          return (
            <button
              key={option}
              type="button"
              onClick={() => toggleOption(option)}
              className={`px-3 py-1.5 rounded-full text-xs border transition-colors ${
                active
                  ? "bg-rent-purple/30 border-rent-purple text-white"
                  : "bg-black/30 border-rent-border text-white/70 hover:text-white"
              }`}
              disabled={isLoading}
            >
              {option}
            </button>
          );
        })}
      </div>

      <div>
        <h4 className="text-white font-semibold text-sm">Required User Inputs</h4>
        <p className="text-white/50 text-xs mt-1">Confirm or edit these input fields for your final app.</p>
      </div>

      <div className="space-y-2">
        {variables.map((variable, index) => (
          <div key={`variable-${index}`} className="flex flex-col gap-1.5 mb-4">
            <label className="text-xs text-gray-400 font-semibold uppercase tracking-wider ml-1">
              {variable.name}
            </label>
            <input
              type="text"
              value={variableValues[variable.name] || ""}
              placeholder={variable.placeholder || "Enter details..."}
              className="w-full bg-[#121018] border border-[#2a2238] rounded-xl px-4 py-3.5 text-sm text-gray-200 placeholder:text-gray-600 focus:outline-none focus:border-[#8b5cf6]/60 focus:bg-[#1a1525] transition-all duration-300 ease-in-out hover:border-[#3b2d50]"
              onChange={(e) => updateVariableValue(variable.name, e.target.value)}
            />
          </div>
        ))}
      </div>

      <button
        type="button"
        onClick={handleSubmit}
        disabled={isLoading}
        className="px-4 py-2 rounded-lg bg-rent-purple text-white text-sm font-medium hover:bg-rent-purple/90 disabled:opacity-60"
      >
        Continue with these settings
      </button>
    </div>
  );
}
