import ChatWindow from "./components/ChatWindow";
import useChat from "./hooks/useChat";

function App() {
  const chat = useChat();

  return (
    <div className="min-h-screen bg-rent-bg text-white">
      <ChatWindow {...chat} />
    </div>
  );
}

export default App;
