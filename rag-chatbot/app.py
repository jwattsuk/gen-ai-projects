"""
Flask Web Interface for RAG Chatbot
Provides a modern web UI for interactive chatbot conversations
"""

import os
import json
from flask import Flask, render_template, request, jsonify, session
from datetime import datetime
from chatbot import RAGChatBot
import uuid

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

# Global chatbot instance
chatbot = None
chatbot_initialized = False

def initialize_chatbot(search_mode="cached"):
    """Initialize the chatbot with error handling"""
    global chatbot, chatbot_initialized
    
    try:
        print(f"Initializing chatbot in {search_mode} mode...")
        chatbot = RAGChatBot(search_mode=search_mode)
        chatbot.load_knowledge_base("embeddings")
        chatbot_initialized = True
        print("Chatbot initialized successfully!")
        return True
    except Exception as e:
        print(f"Failed to initialize chatbot: {e}")
        chatbot_initialized = False
        return False

@app.route('/')
def index():
    """Main chat interface page"""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    global chatbot, chatbot_initialized
    
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'error': 'Empty message'}), 400
        
        # Initialize chatbot if not already done
        if not chatbot_initialized:
            search_mode = data.get('search_mode', 'cached')
            if not initialize_chatbot(search_mode):
                return jsonify({
                    'error': 'Chatbot initialization failed. Please check your configuration.',
                    'details': 'Make sure your .env file is properly configured with Azure OpenAI credentials.'
                }), 500
        
        # Get response from chatbot
        result = chatbot.chat(message)
        
        # Format response for web interface
        response_data = {
            'response': result['response'],
            'sources': result['sources'],
            'search_mode': result['search_mode'],
            'search_info': result['search_info'],
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({
            'error': 'An error occurred while processing your message.',
            'details': str(e)
        }), 500

@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Clear conversation history"""
    global chatbot, chatbot_initialized
    
    try:
        if chatbot_initialized and chatbot:
            chatbot.clear_history()
            return jsonify({'success': True, 'message': 'Conversation history cleared'})
        else:
            return jsonify({'error': 'Chatbot not initialized'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/status')
def status():
    """Get chatbot status"""
    global chatbot_initialized, chatbot
    
    status_info = {
        'initialized': chatbot_initialized,
        'search_mode': chatbot.search_mode if chatbot else None,
        'model': chatbot.model if chatbot else None,
        'timestamp': datetime.now().isoformat()
    }
    
    return jsonify(status_info)

@app.route('/set_mode', methods=['POST'])
def set_mode():
    """Change search mode"""
    global chatbot, chatbot_initialized
    
    try:
        data = request.get_json()
        mode = data.get('mode', 'cached')
        
        if mode not in ['cached', 'cosmosdb']:
            return jsonify({'error': 'Invalid mode. Must be "cached" or "cosmosdb"'}), 400
        
        # Reinitialize with new mode
        if initialize_chatbot(mode):
            return jsonify({
                'success': True, 
                'message': f'Search mode changed to {mode}',
                'mode': mode
            })
        else:
            return jsonify({'error': f'Failed to initialize chatbot in {mode} mode'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template('error.html', 
                         error_code=404, 
                         error_message="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return render_template('error.html', 
                         error_code=500, 
                         error_message="Internal server error"), 500

def create_app(search_mode="cached", debug=False):
    """Application factory function"""
    app.config['DEBUG'] = debug
    
    # Initialize chatbot on startup
    with app.app_context():
        initialize_chatbot(search_mode)
    
    return app

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Chatbot Web Interface")
    parser.add_argument('--mode', choices=['cached', 'cosmosdb'], default='cached',
                       help='Search mode: cached (in-memory) or cosmosdb (direct vector search)')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    
    args = parser.parse_args()
    
    print(f"Starting RAG Chatbot Web Interface")
    print(f"Mode: {args.mode.upper()}")
    print(f"URL: http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop the server")
    
    # Create and run the app
    web_app = create_app(search_mode=args.mode, debug=args.debug)
    web_app.run(host=args.host, port=args.port, debug=args.debug)