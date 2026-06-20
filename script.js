const chatForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');
const chatContainer = document.getElementById('chat-container');
const typingIndicator = document.getElementById('typing-indicator');

// Simulated gibberish responses to represent the untrained character-level model
const gibberishResponses = [
    "ROMEO:\nO thour art a falstaf to the cometh, and the with shall the to be...",
    "What he shall so be the mands of the sir?",
    "KING:\nI have not the world, and thou art a bawd, and a base, the with...",
    "A bear, a boars, and a sours of the pring! Forsooth!",
    "My lord, I am not a stree, but a fald of the thour...",
    "JULIET:\nWherefore art thou roming? The windes are blowth to the mands.",
    "Exeunt.\n\nEnter the KING and a solider of the with...",
    "Nay, I shall not be the sir of the with!"
];

chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const text = userInput.value.trim();
    if (!text) return;

    // 1. Add User Message
    addMessage(text, 'user');
    userInput.value = '';

    // 2. Show loading/predicting state
    typingIndicator.classList.remove('hidden');
    scrollToBottom();

    // 3. Simulate Backend Request Delay
    // Replace this timeout with your actual fetch() call to your backend API
    // Example:
    // 3. Make the Backend Request
    try {
        // const response = await fetch('http://127.0.0.1:8000/api/chat',
        const response = await fetch('https://vaaakhim-mini-gpt-shakespeare.hf.space/api/chat',
        {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json' // Crucial: Tell FastAPI to expect JSON
            },
            // Change "prompt" to "message" to match your Pydantic ChatRequest model
            body: JSON.stringify({ message: text })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        // Change "data.text" to "data.reply" to match your Pydantic ChatResponse model
        const generatedText = data.reply;

        // 4. Handle the UI updates
        typingIndicator.classList.add('hidden');

        // Stream the actual model response instead of the gibberish array
        streamBotMessage(generatedText);

    } catch (error) {
        console.error("Failed to connect to the chatbot API:", error);
        typingIndicator.classList.add('hidden');

        // Optional: Show a graceful error message in the chat UI
        streamBotMessage("Sorry, I'm having trouble connecting to the server right now.");
    }
});

function addMessage(text, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `flex flex-col gap-1 max-w-[80%] ${sender === 'user' ? 'self-end items-end' : 'self-start items-start'}`;

    const bubble = document.createElement('div');

    if (sender === 'user') {
        bubble.className = 'bg-zinc-800 text-white px-5 py-3 rounded-2xl rounded-tr-sm shadow-sm text-sm';
        bubble.textContent = text;

        const label = document.createElement('span');
        label.className = 'text-[10px] text-zinc-400 mr-1';
        label.textContent = 'You';

        messageDiv.appendChild(bubble);
        messageDiv.appendChild(label);
    } else {
        // This branch is used for instant bot messages if needed,
        // but we primarily use streamBotMessage for the bot.
        bubble.className = 'bg-zinc-100 border border-zinc-200 text-zinc-800 px-5 py-3 rounded-2xl rounded-tl-sm shadow-sm shakespeare-font text-lg whitespace-pre-wrap';
        bubble.textContent = text;

        const label = document.createElement('span');
        label.className = 'text-[10px] text-zinc-400 ml-1';
        label.textContent = 'The Bard • Simulated';

        messageDiv.appendChild(bubble);
        messageDiv.appendChild(label);
    }

    chatContainer.appendChild(messageDiv);
    scrollToBottom();
}

// Simulates the character-by-character prediction of your mini-GPT
function streamBotMessage(fullText) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `flex flex-col gap-1 max-w-[80%] self-start items-start`;

    const bubble = document.createElement('div');
    // Adding 'typing-cursor' class for the blinking cursor effect
    bubble.className = 'bg-zinc-100 border border-zinc-200 text-zinc-800 px-5 py-3 rounded-2xl rounded-tl-sm shadow-sm shakespeare-font text-lg whitespace-pre-wrap typing-cursor transition-all';

    const label = document.createElement('span');
    label.className = 'text-[10px] text-zinc-400 ml-1';
    label.textContent = 'The Bard • Character Gen';

    messageDiv.appendChild(bubble);
    messageDiv.appendChild(label);
    chatContainer.appendChild(messageDiv);

    let i = 0;
    // The interval speed (e.g., 30ms) simulates how fast the transformer predicts the next character
    const typingInterval = setInterval(() => {
        if (i < fullText.length) {
            bubble.textContent += fullText.charAt(i);
            i++;
            scrollToBottom();
        } else {
            clearInterval(typingInterval);
            // Remove cursor when done
            bubble.classList.remove('typing-cursor');
        }
    }, 30); // 30ms per character - adjust to match your model's inference speed
}

function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
}