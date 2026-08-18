import { useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000/chat";

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [conversationId, setConversationId] = useState(null);
  const [loading, setLoading] = useState(false);

  const sendMessage = async (text = input) => {
    const message = text.trim();

    if (!message || loading) {
      return;
    }

    setInput("");

    setMessages((prev) => [
      ...prev,
      {
        role: "user",
        content: message,
      },
    ]);

    setLoading(true);

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message,
          conversation_id: conversationId,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json();

      if (data.conversation_id) {
        setConversationId(data.conversation_id);
      }

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content:
            data.answer || "I could not determine an answer.",
          toolCalls: data.tool_calls || [],
          executionTime: Number(data.execution_time || 0),
        },
      ]);
    } catch (error) {
      console.error("KUDOO API error:", error);

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content:
            "Sorry, I could not connect to KUDOO.",
          error: error.message,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  };

  const newChat = () => {
    setMessages([]);
    setConversationId(null);
    setInput("");
  };

  return (
    <div className="app">

      {/* HEADER */}
      <header className="header">
        <div>
          <div className="brand">
            <span className="brand-icon">✦</span>
            <span>KUDOO</span>
          </div>

          <div className="subtitle">
            AI-powered sales data analyst
          </div>
        </div>

        <button
          className="new-chat"
          onClick={newChat}
          disabled={loading}
        >
          + New chat
        </button>
      </header>


      {/* CHAT */}
      <main className="chat-container">

        {messages.length === 0 && (
          <div className="empty-state">

            <div className="empty-icon">
              ✦
            </div>

            <h1>Ask KUDOO</h1>

            <p>
              Ask questions about your sales data using
              natural language.
            </p>

            <div className="suggestions">

              <button
                onClick={() =>
                  sendMessage(
                    "Which product made the most money?"
                  )
                }
              >
                Which product made the most money?
              </button>

              <button
                onClick={() =>
                  sendMessage(
                    "What percentage of total revenue did the Laptop contribute?"
                  )
                }
              >
                What percentage of total revenue did the
                Laptop contribute?
              </button>

              <button
                onClick={() =>
                  sendMessage(
                    "How much more did the Laptop make than the Phone?"
                  )
                }
              >
                How much more did the Laptop make than the
                Phone?
              </button>

            </div>
          </div>
        )}


        {/* MESSAGES */}
        {messages.map((message, index) => (
          <div
            className={`message-row ${message.role}`}
            key={index}
          >

            {message.role === "assistant" && (
              <div className="avatar">
                ✦
              </div>
            )}


            <div className="message-content">

              <div className="message-bubble">
                {message.content}
              </div>


              {message.error && (
                <div className="error-details">
                  {message.error}
                </div>
              )}


              {message.role === "assistant" &&
                message.toolCalls?.length > 0 && (
                  <AnalysisDetails
                    toolCalls={message.toolCalls}
                    executionTime={message.executionTime}
                  />
                )}

            </div>


            {message.role === "user" && (
              <div className="user-label">
                You
              </div>
            )}

          </div>
        ))}


        {/* LOADING */}
        {loading && (
          <div className="message-row assistant">

            <div className="avatar">
              ✦
            </div>

            <div className="message-bubble typing">
              <span className="typing-dot">●</span>
              <span className="typing-dot">●</span>
              <span className="typing-dot">●</span>
              <span className="typing-text">
                KUDOO is analyzing your data...
              </span>
            </div>

          </div>
        )}

      </main>


      {/* INPUT */}
      <footer className="footer">

        <div className="input-container">

          <textarea
            value={input}
            onChange={(event) =>
              setInput(event.target.value)
            }
            onKeyDown={handleKeyDown}
            placeholder="Ask KUDOO about your sales data..."
            rows={1}
            disabled={loading}
          />

          <button
            onClick={() => sendMessage()}
            disabled={loading || !input.trim()}
          >
            {loading ? "..." : "Send"}
          </button>

        </div>


        <div className="footer-info">

          <span>
            {conversationId
              ? "● Conversation memory active"
              : "○ New conversation"}
          </span>

          <span>
            KUDOO · AI Data Analyst
          </span>

        </div>

      </footer>

    </div>
  );
}


/* ==================================================
   ANALYSIS DETAILS
================================================== */

function AnalysisDetails({
  toolCalls,
  executionTime,
}) {
  return (
    <details className="analysis-details">

      <summary>
        <span>✓</span>
        Analysis details
      </summary>

      <div className="analysis-body">

        {toolCalls.map((call, index) => (
          <div
            className="tool-call"
            key={index}
          >

            <div className="tool-header">

              <span className="tool-status">
                ✓
              </span>

              <strong>
                {call.tool}
              </strong>

              <span className="tool-time">
                {(
                  Number(call.duration || 0) * 1000
                ).toFixed(0)}{" "}
                ms
              </span>

            </div>


            <ToolResult result={call.result} />

          </div>
        ))}


        <div className="execution-total">
          Total response time:{" "}
          <strong>
            {Number(executionTime).toFixed(2)} sec
          </strong>
        </div>

      </div>

    </details>
  );
}


/* ==================================================
   TOOL RESULT
================================================== */

function ToolResult({ result }) {
  if (!result) {
    return null;
  }

  return (
    <div className="tool-result">

      {result.product !== undefined && (
        <div>
          Product:{" "}
          <strong>
            {result.product}
          </strong>
        </div>
      )}

      {result.revenue !== undefined && (
        <div>
          Revenue:{" "}
          <strong>
            $
            {Number(result.revenue).toLocaleString()}
          </strong>
        </div>
      )}

      {result.percentage !== undefined && (
        <div>
          Percentage:{" "}
          <strong>
            {Number(result.percentage).toFixed(1)}%
          </strong>
        </div>
      )}

      {result.difference !== undefined && (
        <div>
          Difference:{" "}
          <strong>
            $
            {Number(result.difference).toLocaleString()}
          </strong>
        </div>
      )}

      {result.total_revenue !== undefined && (
        <div>
          Total revenue:{" "}
          <strong>
            $
            {Number(result.total_revenue).toLocaleString()}
          </strong>
        </div>
      )}

      {result.message !== undefined && (
        <div>
          {result.message}
        </div>
      )}

    </div>
  );
}

export default App;