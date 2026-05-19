const chatInput = document.getElementById('chat-input');
const sendButton = document.getElementById('send-button');
const chatMessages = document.getElementById('chat-messages');

function autoResize(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = (textarea.scrollHeight < 200 ? textarea.scrollHeight : 200) + 'px';

    // Enable/disable send button based on content
    if(textarea.value.trim() !== '') {
        sendButton.removeAttribute('disabled');
    } else {
        sendButton.setAttribute('disabled', 'true');
    }
}

// Initial check
autoResize(chatInput);

function setInput(text) {
    chatInput.value = text;
    autoResize(chatInput);
    chatInput.focus();
}

function appendMessage(role, text) {
    // Remove welcome state if it exists
    const welcome = document.querySelector('.welcome-state');
    if (welcome) welcome.remove();

    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;

    const isUser = role === 'user';
    const svgIcon = isUser
        ? '<svg viewBox="0 0 16 16" width="16" height="16" fill="currentColor"><path d="M8 1.5a4 4 0 1 0 0 8 4 4 0 0 0 0-8zM3 5.5a5 5 0 1 1 10 0 5 5 0 0 1-10 0z"/><path d="M12.25 11.25a.75.75 0 0 0-.75.75 3.5 3.5 0 0 1-7 0 .75.75 0 0 0-1.5 0 5 5 0 0 0 10 0 .75.75 0 0 0-.75-.75z"/></svg>'
        : '<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12 12-5.4 12-12S18.66 0 12 0zm5.521 17.34c-.24.359-.66.48-1.021.24-2.82-1.74-6.36-2.101-10.561-1.141-.418.122-.779-.179-.899-.539-.12-.421.18-.78.54-.9 4.56-1.021 8.52-.6 11.64 1.32.42.18.54.659.301 1.02zm1.44-3.3c-.301.42-.841.6-1.262.3-3.239-1.98-8.159-2.58-11.939-1.38-.479.12-1.02-.12-1.14-.6-.12-.48.12-1.021.6-1.141C9.6 9.9 15 10.561 18.72 12.84c.361.181.54.78.241 1.2zm.12-3.36C15.24 8.4 8.82 8.16 5.16 9.301c-.6.18-1.2-.181-1.38-.781-.18-.6.18-1.2.78-1.381 4.26-1.26 11.28-1.02 15.721 1.621.539.3.719 1.02.419 1.56-.299.421-1.02.599-1.559.3z"/></svg>';

    // Simple markdown parsing for the AI
    let formattedText = text;
    if(!isUser) {
        // Basic code block support
        formattedText = text.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
        // Basic bold
        formattedText = formattedText.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        // Basic line breaks
        formattedText = formattedText.replace(/\n/g, '<br>');
    }

    msgDiv.innerHTML = `
        <div class="message-avatar">${svgIcon}</div>
        <div class="message-content">${formattedText}</div>
    `;

    chatMessages.appendChild(msgDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    return msgDiv.querySelector('.message-content');
}

async function handleSend() {
    const message = chatInput.value.trim();
    if (!message) return;

    // UI Updates
    appendMessage('user', message);
    chatInput.value = '';
    autoResize(chatInput);

    // Add temporary AI loading message
    const aiContentDiv = appendMessage('ai', '...');

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message })
        });

        if (!response.ok) throw new Error('Network response was not ok');

        // Setup SSE reader
        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let aiMessageText = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });

            // Parse SSE format "data: <content>\n\n"
            const lines = chunk.split('\n');
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = line.slice(6);
                    if(data.startsWith('Error:')) {
                         aiMessageText = data;
                    } else {
                         aiMessageText += data;
                    }

                    // Simple formatting update
                    let formattedText = aiMessageText;
                    formattedText = formattedText.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
                    formattedText = formattedText.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
                    formattedText = formattedText.replace(/\n/g, '<br>');

                    aiContentDiv.innerHTML = formattedText;
                    chatMessages.scrollTop = chatMessages.scrollHeight;
                }
            }
        }
    } catch (error) {
        aiContentDiv.innerHTML = `<span style="color: var(--text-negative)">Error: ${error.message}</span>`;
    }
}

sendButton.addEventListener('click', handleSend);
chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
    }
});
