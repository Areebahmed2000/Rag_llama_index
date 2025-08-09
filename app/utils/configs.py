prompt = """You are a precise Q&A system that provides accurate answers from provided documents.

INSTRUCTIONS:
- Provide accurate answers based on the information in the provided context
- Always cite your sources with file names and page numbers
- If you cannot find relevant information in the provided context, respond with: "No information available in our RAG system."
- Be precise and factual - only use information directly from the provided context
- just give me answer to the user as it is present in documnet.
- dont change anything in answer
Remember: Accuracy is the priority. Only provide information that is supported by the documents."""



html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Agentic RAG Chat</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
            .header { background: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); text-align: center; }
            .upload-section { background: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .chat-container { display: flex; gap: 20px; height: 600px; }
            .chat-messages { flex: 1; background: white; border-radius: 10px; padding: 20px; overflow-y: auto; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .sources-panel { width: 350px; background: white; border-radius: 10px; padding: 20px; overflow-y: auto; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .message { margin-bottom: 15px; padding: 15px; border-radius: 10px; }
            .user-message { background: linear-gradient(135deg, #007bff, #0056b3); color: white; margin-left: 50px; }
            .assistant-message { background: #f8f9fa; border: 2px solid #e9ecef; margin-right: 50px; }
            .no-answer { background: #fff3cd; border: 2px solid #ffeaa7; color: #856404; }
            .input-section { background: white; padding: 20px; border-radius: 10px; margin-top: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .input-group { display: flex; gap: 10px; margin-bottom: 10px; }
            input[type="text"] { flex: 1; padding: 15px; border: 2px solid #ddd; border-radius: 8px; font-size: 16px; }
            input[type="text"]:focus { outline: none; border-color: #007bff; }
            button { padding: 15px 25px; background: #007bff; color: white; border: none; border-radius: 8px; cursor: pointer; font-weight: bold; }
            button:hover { background: #0056b3; }
            button:disabled { background: #ccc; cursor: not-allowed; }
            .file-input { margin-bottom: 15px; padding: 15px; border: 2px dashed #ddd; border-radius: 8px; text-align: center; }
            .source-item { background: #f8f9fa; padding: 15px; margin-bottom: 10px; border-radius: 8px; border-left: 4px solid #007bff; }
            .citation-header { background: #007bff; color: white; padding: 10px; margin: -20px -20px 20px -20px; border-radius: 10px 10px 0 0; }
            .status { padding: 15px; margin: 10px 0; border-radius: 8px; font-weight: bold; }
            .status.success { background: #d4edda; color: #155724; border: 2px solid #c3e6cb; }
            .status.error { background: #f8d7da; color: #721c24; border: 2px solid #f5c6cb; }
            .status.info { background: #d1ecf1; color: #0c5460; border: 2px solid #bee5eb; }
            .loading { display: none; text-align: center; padding: 30px; }
            .controls { display: flex; justify-content: space-between; align-items: center; }
            .agent-toggle { margin-left: 10px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 Agentic RAG Application</h1>
                <p>Upload documents and ask questions with conversation memory and citations</p>
            </div>

            <div class="upload-section">
                <h3>📁 Document Upload</h3>
                <div class="file-input">
                    <input type="file" id="fileInput" multiple accept=".pdf,.csv,.xlsx,.xls">
                    <br><br>
                    <button onclick="uploadFiles()" style="background: #28a745;">📤 Upload & Process Documents</button>
                </div>
                <div id="uploadStatus"></div>
            </div>

            <div class="chat-container">
                <div class="chat-messages">
                    <h3>💬 Chat</h3>
                    <div id="messages"></div>
                    <div class="loading" id="loading">
                        <p>🤔 Thinking...</p>
                    </div>
                </div>
                
                <div class="sources-panel">
                    <div class="citation-header">
                        <h3>📚 Sources & Citations</h3>
                    </div>
                    <div id="sources">
                        <p><em>Citations will appear here when you ask questions</em></p>
                    </div>
                </div>
            </div>

            <div class="input-section">
                <div class="input-group">
                    <input type="text" id="questionInput" placeholder="💭 Ask a question about your uploaded documents..." onkeypress="if(event.key==='Enter') askQuestion()">
                    <button onclick="askQuestion()" id="askBtn">🚀 Ask Question</button>
                </div>
                <div class="controls">
                    <div class="agent-toggle">
                        <label>
                            <input type="checkbox" id="useAgent" checked> 🤖 Use Agentic Reasoning (multi-step)
                        </label>
                    </div>
                    <button onclick="clearChat()" style="background: #dc3545;">🗑️ Clear Chat</button>
                </div>
            </div>
        </div>

        <script>
            let conversationId = 0;

            async function uploadFiles() {
                const fileInput = document.getElementById('fileInput');
                const statusDiv = document.getElementById('uploadStatus');
                
                if (fileInput.files.length === 0) {
                    showStatus('Please select files to upload', 'error');
                    return;
                }

                const formData = new FormData();
                for (let file of fileInput.files) {
                    formData.append('files', file);
                }

                try {
                    showStatus('Uploading and processing files...', 'info');
                    const response = await fetch('/upload_documents/', {
                        method: 'POST',
                        body: formData
                    });

                    const result = await response.json();
                    if (response.ok) {
                        showStatus(result.message, 'success');
                        fileInput.value = '';
                    } else {
                        showStatus(`Error: ${result.detail}`, 'error');
                    }
                } catch (error) {
                    showStatus(`Upload failed: ${error.message}`, 'error');
                }
            }

            async function askQuestion() {
                const questionInput = document.getElementById('questionInput');
                const question = questionInput.value.trim();
                const useAgent = document.getElementById('useAgent').checked;
                
                if (!question) return;

                // Add user message
                addMessage(question, 'user');
                questionInput.value = '';
                
                // Show loading
                document.getElementById('loading').style.display = 'block';
                document.getElementById('askBtn').disabled = true;

                try {
                    const response = await fetch('/chat/', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ 
                            question: question,
                            use_agent: useAgent 
                        })
                    });

                    const result = await response.json();
                    
                    if (response.ok) {
                        addMessage(result.answer, 'assistant');
                        showSources(result.sources);
                        conversationId = result.conversation_id;
                    } else {
                        addMessage(`Error: ${result.detail}`, 'assistant');
                    }
                } catch (error) {
                    addMessage(`Failed to get response: ${error.message}`, 'assistant');
                } finally {
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('askBtn').disabled = false;
                }
            }

            function addMessage(content, role) {
                const messagesDiv = document.getElementById('messages');
                const messageDiv = document.createElement('div');
                
                // Check if it's a "no answer found" response
                const isNoAnswer = content.toLowerCase().includes('no information available');
                
                let messageClass = `message ${role}-message`;
                if (isNoAnswer && role === 'assistant') {
                    messageClass += ' no-answer';
                }
                
                messageDiv.className = messageClass;
                
                const icon = role === 'user' ? '👤' : (isNoAnswer ? '🚫' : '🤖');
                const label = role === 'user' ? 'You' : 'Assistant';
                
                messageDiv.innerHTML = `<strong>${icon} ${label}:</strong><br>${content}`;
                messagesDiv.appendChild(messageDiv);
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }

            function showSources(sources) {
                const sourcesDiv = document.getElementById('sources');
                if (sources && sources.length > 0) {
                    sourcesDiv.innerHTML = '<h4>📖 Document Citations:</h4>';
                    sources.forEach((source, index) => {
                        const sourceDiv = document.createElement('div');
                        sourceDiv.className = 'source-item';
                        
                        let matchTypeIcon = '';
                        let matchTypeText = '';
                        
                        if (source.match_type === 'exact') {
                            matchTypeIcon = '🎯';
                            matchTypeText = 'Exact Question Match';
                            sourceDiv.style.borderLeft = '4px solid #28a745'; // Green for exact
                        } else if (source.match_type === 'fuzzy_exact') {
                            matchTypeIcon = '🔍';
                            matchTypeText = 'Near-Exact Question Match';
                            sourceDiv.style.borderLeft = '4px solid #17a2b8'; // Cyan for fuzzy exact
                        } else if (source.match_type === 'semantic_high') {
                            matchTypeIcon = '🧠';
                            matchTypeText = 'High-Similarity Semantic Match';
                            sourceDiv.style.borderLeft = '4px solid #ffc107'; // Yellow for semantic
                        } else {
                            matchTypeIcon = '🔎';
                            matchTypeText = 'General Search';
                            sourceDiv.style.borderLeft = '4px solid #007bff'; // Blue for general
                        }
                        
                        sourceDiv.innerHTML = `
                            <strong>📑 Citation ${index + 1}: ${matchTypeIcon} ${matchTypeText}</strong><br>
                            <strong>📄 File:</strong> ${source.file_name}<br>
                            <strong>📝 Page:</strong> ${source.page_number}<br>
                            <strong>📊 Type:</strong> ${source.document_type}
                            ${source.similarity_score ? `<br><strong>🎯 Match Score:</strong> ${(source.similarity_score * 100).toFixed(1)}%` : ''}
                            ${source.original_question ? `<br><strong>❓ Original Q:</strong> <em>${source.original_question.substring(0, 100)}...</em>` : ''}
                        `;
                        sourcesDiv.appendChild(sourceDiv);
                    });
                } else {
                    sourcesDiv.innerHTML = '<p><em>❌ No sources found for the last response</em></p>';
                }
            }

            async function clearChat() {
                try {
                    await fetch('/clear_conversation/', { method: 'POST' });
                    document.getElementById('messages').innerHTML = '';
                    document.getElementById('sources').innerHTML = '<p><em>Citations will appear here when you ask questions</em></p>';
                    conversationId = 0;
                    showStatus('Conversation cleared', 'success');
                } catch (error) {
                    showStatus(`Failed to clear conversation: ${error.message}`, 'error');
                }
            }

            function showStatus(message, type) {
                const statusDiv = document.getElementById('uploadStatus');
                statusDiv.className = `status ${type}`;
                statusDiv.innerHTML = `<strong>${type === 'success' ? '✅' : type === 'error' ? '❌' : 'ℹ️'}</strong> ${message}`;
                setTimeout(() => {
                    statusDiv.textContent = '';
                    statusDiv.className = '';
                }, 5000);
            }
        </script>
    </body>
    </html>
    """