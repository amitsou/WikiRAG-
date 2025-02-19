async function sendQuery() {
    let userQuery = document.getElementById("userQuery").value.trim();
    if (!userQuery) return;

    let chatBox = document.getElementById("chatBox");

    // Display user message
    let userMsg = document.createElement("div");
    userMsg.className = "user-message message";
    userMsg.innerText = userQuery;
    chatBox.appendChild(userMsg);
    chatBox.scrollTop = chatBox.scrollHeight;

    // Display typing indicator
    let typingIndicator = document.createElement("div");
    typingIndicator.className = "bot-message message typing-indicator";
    typingIndicator.innerText = "Assistant is typing...";
    chatBox.appendChild(typingIndicator);
    chatBox.scrollTop = chatBox.scrollHeight;

    // Clear input field
    document.getElementById("userQuery").value = "";

    try {
        let response = await fetch("http://127.0.0.1:8000/query", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: userQuery })
        });

        let result = await response.json();
        let botReply = result.response || "Sorry, I couldn't process that.";

        // Remove typing indicator
        chatBox.removeChild(typingIndicator);

        // Extract only the LLM's response by removing unnecessary context
        let botReplyText = botReply.replace(/Context:.*?User Query:.*?Based on the above context, provide a helpful response./s, "").trim();

        // Display only the LLM's response
        let botMsg = document.createElement("div");
        botMsg.className = "bot-message message";
        botMsg.innerText = botReplyText;
        chatBox.appendChild(botMsg);

        chatBox.scrollTop = chatBox.scrollHeight;

    } catch (error) {
        chatBox.removeChild(typingIndicator);
        let errorMsg = document.createElement("div");
        errorMsg.className = "bot-message message";
        errorMsg.innerText = "Error: Unable to reach the API.";
        chatBox.appendChild(errorMsg);
    }
}
