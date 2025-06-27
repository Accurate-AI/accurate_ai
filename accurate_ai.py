#!/usr/bin/env python3
"""
Advanced AI Assistant Application
A comprehensive AI tool with GUI, terminal, and web search capabilities
Built with discrete mathematics, ML, DL, and neural network principles
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog, simpledialog
import threading
import queue
import json
import os
import sys
import subprocess
import webbrowser
import requests
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sqlite3
import hashlib
import re
import time
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration and Constants
CONFIG_FILE = "ai_assistant_config.json"
DATABASE_FILE = "ai_assistant.db"
VERSION = "1.0.0"

# Color Theme
COLORS = {
    'primary_red': '#DC143C',
    'primary_yellow': '#FFD700',
    'dark_red': '#8B0000',
    'light_yellow': '#FFFFE0',
    'white': '#FFFFFF',
    'black': '#000000',
    'gray': '#808080',
    'light_gray': '#D3D3D3'
}

@dataclass
class AIResponse:
    """Data class for AI responses"""
    question: str
    answer: str
    confidence: float
    timestamp: datetime
    source: str

class DiscreteMatrixProcessor:
    """Discrete mathematics processor for logical operations"""
    
    def __init__(self):
        self.logic_operators = {
            'AND': self._and_operation,
            'OR': self._or_operation,
            'NOT': self._not_operation,
            'XOR': self._xor_operation,
            'IMPLIES': self._implies_operation
        }
    
    def _and_operation(self, a, b):
        return a and b
    
    def _or_operation(self, a, b):
        return a or b
    
    def _not_operation(self, a, b=None):
        return not a
    
    def _xor_operation(self, a, b):
        return a != b
    
    def _implies_operation(self, a, b):
        return not a or b
    
    def evaluate_logical_expression(self, expression: str) -> bool:
        """Evaluate logical expressions using discrete mathematics principles"""
        try:
            # Simple evaluation for demonstration
            expression = expression.upper()
            if 'TRUE' in expression or 'T' in expression:
                return True
            elif 'FALSE' in expression or 'F' in expression:
                return False
            return bool(eval(expression.lower()))
        except:
            return False
    
    def generate_truth_table(self, variables: List[str]) -> Dict:
        """Generate truth table for given variables"""
        n = len(variables)
        combinations = []
        for i in range(2**n):
            combination = []
            for j in range(n):
                combination.append(bool(i & (1 << j)))
            combinations.append(combination)
        return {'variables': variables, 'combinations': combinations}

class NeuralNetworkSimulator:
    """Simple neural network simulator for educational purposes"""
    
    def __init__(self, layers: List[int]):
        self.layers = layers
        self.weights = []
        self.biases = []
        self._initialize_network()
    
    def _initialize_network(self):
        """Initialize network weights and biases"""
        for i in range(len(self.layers) - 1):
            weight_matrix = np.random.randn(self.layers[i], self.layers[i+1]) * 0.1
            bias_vector = np.zeros((1, self.layers[i+1]))
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    def forward_pass(self, input_data):
        """Perform forward pass through the network"""
        activation = input_data
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(activation, weight) + bias
            activation = self.sigmoid(z)
        return activation
    
    def train_step(self, input_data, target_data, learning_rate=0.01):
        """Perform one training step (simplified backpropagation)"""
        # Forward pass
        activations = [input_data]
        for weight, bias in zip(self.weights, self.biases):
            z = np.dot(activations[-1], weight) + bias
            activation = self.sigmoid(z)
            activations.append(activation)
        
        # Simplified backward pass
        error = target_data - activations[-1]
        for i in range(len(self.weights) - 1, -1, -1):
            delta = error * activations[i+1] * (1 - activations[i+1])
            self.weights[i] += learning_rate * np.dot(activations[i].T, delta)
            self.biases[i] += learning_rate * np.sum(delta, axis=0, keepdims=True)
            error = np.dot(delta, self.weights[i].T)
        
        return np.mean(np.square(target_data - activations[-1]))

class MachineLearningProcessor:
    """Machine learning algorithms processor"""
    
    def __init__(self):
        self.models = {}
    
    def linear_regression(self, x_data, y_data):
        """Simple linear regression implementation"""
        n = len(x_data)
        sum_x = sum(x_data)
        sum_y = sum(y_data)
        sum_xy = sum(x * y for x, y in zip(x_data, y_data))
        sum_x2 = sum(x * x for x in x_data)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / n
        
        return slope, intercept
    
    def k_means_clustering(self, data, k=3, max_iterations=100):
        """K-means clustering implementation"""
        if not data:
            return []
        
        # Initialize centroids randomly
        centroids = random.sample(data, k)
        
        for _ in range(max_iterations):
            clusters = [[] for _ in range(k)]
            
            # Assign points to nearest centroid
            for point in data:
                distances = [sum((point[i] - centroid[i])**2 for i in range(len(point)))**0.5 
                           for centroid in centroids]
                cluster_index = distances.index(min(distances))
                clusters[cluster_index].append(point)
            
            # Update centroids
            new_centroids = []
            for cluster in clusters:
                if cluster:
                    centroid = [sum(point[i] for point in cluster) / len(cluster) 
                              for i in range(len(cluster[0]))]
                    new_centroids.append(centroid)
                else:
                    new_centroids.append(centroids[len(new_centroids)])
            
            if new_centroids == centroids:
                break
            centroids = new_centroids
        
        return centroids, clusters

class WebSearchEngine:
    """Web search functionality"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.search_engines = {
            'duckduckgo': self._search_duckduckgo,
            'google': self._search_google,
            'bing': self._search_bing
        }
    
    def _search_duckduckgo(self, query: str, max_results: int = 5):
        """Search using DuckDuckGo (free alternative)"""
        try:
            url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1&skip_disambig=1"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            results = []
            if 'RelatedTopics' in data:
                for item in data['RelatedTopics'][:max_results]:
                    if 'Text' in item:
                        results.append({
                            'title': item.get('Text', '')[:100],
                            'url': item.get('FirstURL', ''),
                            'snippet': item.get('Text', '')
                        })
            
            return results
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return []
    
    def _search_google(self, query: str, max_results: int = 5):
        """Search using Google Custom Search API"""
        if not self.api_key:
            return []
        
        try:
            # This would require Google Custom Search API key and CX
            # Placeholder implementation
            return [{'title': 'Google search requires API key', 'url': '', 'snippet': 'Configure API key in settings'}]
        except Exception as e:
            logger.error(f"Google search error: {e}")
            return []
    
    def _search_bing(self, query: str, max_results: int = 5):
        """Search using Bing Search API"""
        if not self.api_key:
            return []
        
        try:
            # Placeholder for Bing API implementation
            return [{'title': 'Bing search requires API key', 'url': '', 'snippet': 'Configure API key in settings'}]
        except Exception as e:
            logger.error(f"Bing search error: {e}")
            return []
    
    def search(self, query: str, engine: str = 'duckduckgo', max_results: int = 5):
        """Perform web search using specified engine"""
        if engine in self.search_engines:
            return self.search_engines[engine](query, max_results)
        return []

class DatabaseManager:
    """Database management for storing conversations and data"""
    
    def __init__(self, db_file: str = DATABASE_FILE):
        self.db_file = db_file
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                confidence REAL,
                source TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_conversation(self, response: AIResponse):
        """Save conversation to database"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO conversations (question, answer, confidence, source, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (response.question, response.answer, response.confidence, response.source, response.timestamp))
        
        conn.commit()
        conn.close()
    
    def get_conversation_history(self, limit: int = 100):
        """Retrieve conversation history"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT question, answer, confidence, source, timestamp
            FROM conversations
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [AIResponse(q, a, c, datetime.fromisoformat(t), s) for q, a, c, s, t in rows]
    
    def save_setting(self, key: str, value: str):
        """Save setting to database"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO settings (key, value)
            VALUES (?, ?)
        ''', (key, value))
        
        conn.commit()
        conn.close()
    
    def get_setting(self, key: str, default: str = None):
        """Get setting from database"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute('SELECT value FROM settings WHERE key = ?', (key,))
        row = cursor.fetchone()
        conn.close()
        
        return row[0] if row else default

class AICore:
    """Core AI processing engine"""
    
    def __init__(self):
        self.discrete_processor = DiscreteMatrixProcessor()
        self.neural_network = NeuralNetworkSimulator([10, 5, 3, 1])
        self.ml_processor = MachineLearningProcessor()
        self.web_search = WebSearchEngine()
        self.knowledge_base = self._initialize_knowledge_base()
    
    def _initialize_knowledge_base(self):
        """Initialize basic knowledge base"""
        return {
            'greetings': ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening'],
            'farewells': ['goodbye', 'bye', 'see you', 'farewell'],
            'math_keywords': ['calculate', 'compute', 'solve', 'equation', 'formula'],
            'logic_keywords': ['logic', 'boolean', 'true', 'false', 'and', 'or', 'not'],
            'ml_keywords': ['machine learning', 'neural network', 'training', 'prediction'],
            'search_keywords': ['search', 'find', 'look up', 'web', 'internet']
        }
    
    def process_question(self, question: str, use_web_search: bool = False) -> AIResponse:
        """Process user question and generate AI response"""
        question_lower = question.lower()
        confidence = 0.7
        source = "AI Core"
        
        # Determine question type and generate appropriate response
        if any(keyword in question_lower for keyword in self.knowledge_base['greetings']):
            answer = self._generate_greeting_response()
            confidence = 0.9
        
        elif any(keyword in question_lower for keyword in self.knowledge_base['math_keywords']):
            answer = self._process_math_question(question)
            confidence = 0.8
            source = "Mathematical Processor"
        
        elif any(keyword in question_lower for keyword in self.knowledge_base['logic_keywords']):
            answer = self._process_logic_question(question)
            confidence = 0.8
            source = "Discrete Mathematics Processor"
        
        elif any(keyword in question_lower for keyword in self.knowledge_base['ml_keywords']):
            answer = self._process_ml_question(question)
            confidence = 0.7
            source = "Machine Learning Processor"
        
        elif use_web_search or any(keyword in question_lower for keyword in self.knowledge_base['search_keywords']):
            answer = self._process_web_search_question(question)
            confidence = 0.6
            source = "Web Search Engine"
        
        else:
            answer = self._generate_general_response(question)
            confidence = 0.5
            source = "General AI Processor"
        
        return AIResponse(question, answer, confidence, datetime.now(), source)
    
    def _generate_greeting_response(self):
        """Generate greeting response"""
        responses = [
            "Hello! I'm your AI assistant. How can I help you today?",
            "Hi there! I'm ready to assist you with any questions.",
            "Greetings! I'm here to help with AI, mathematics, logic, and more.",
            "Hello! Feel free to ask me anything about AI, ML, or any other topic."
        ]
        return random.choice(responses)
    
    def _process_math_question(self, question: str):
        """Process mathematical questions"""
        try:
            # Extract numbers and operations
            numbers = re.findall(r'-?\d+\.?\d*', question)
            if len(numbers) >= 2:
                a, b = float(numbers[0]), float(numbers[1])
                
                if '+' in question or 'add' in question or 'plus' in question:
                    result = a + b
                    return f"The sum of {a} and {b} is {result}."
                elif '-' in question or 'subtract' in question or 'minus' in question:
                    result = a - b
                    return f"The difference between {a} and {b} is {result}."
                elif '*' in question or 'multiply' in question or 'times' in question:
                    result = a * b
                    return f"The product of {a} and {b} is {result}."
                elif '/' in question or 'divide' in question:
                    if b != 0:
                        result = a / b
                        return f"The quotient of {a} divided by {b} is {result}."
                    else:
                        return "Error: Division by zero is undefined."
            
            return "I can help with basic arithmetic operations. Please provide numbers and an operation (+, -, *, /)."
        
        except Exception as e:
            return f"I encountered an error processing your math question: {str(e)}"
    
    def _process_logic_question(self, question: str):
        """Process logical and discrete mathematics questions"""
        try:
            if 'truth table' in question.lower():
                variables = re.findall(r'\b[A-Z]\b', question)
                if variables:
                    truth_table = self.discrete_processor.generate_truth_table(variables[:3])
                    return f"Generated truth table for variables: {', '.join(truth_table['variables'])}"
                else:
                    return "Please specify variables (e.g., A, B, C) for the truth table."
            
            elif any(op in question.upper() for op in ['AND', 'OR', 'NOT', 'XOR']):
                result = self.discrete_processor.evaluate_logical_expression(question)
                return f"The logical expression evaluates to: {result}"
            
            else:
                return "I can help with logical operations (AND, OR, NOT, XOR) and truth tables. Please specify your logical expression."
        
        except Exception as e:
            return f"Error processing logic question: {str(e)}"
    
    def _process_ml_question(self, question: str):
        """Process machine learning questions"""
        try:
            if 'neural network' in question.lower():
                return "Neural networks are computational models inspired by biological neural networks. They consist of interconnected nodes (neurons) organized in layers that process information through weighted connections."
            
            elif 'machine learning' in question.lower():
                return "Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed. It includes supervised, unsupervised, and reinforcement learning."
            
            elif 'training' in question.lower():
                # Demonstrate neural network training
                input_data = np.random.randn(1, 10)
                target_data = np.random.randn(1, 1)
                error = self.neural_network.train_step(input_data, target_data)
                return f"Neural network training step completed. Current error: {error:.4f}"
            
            elif 'prediction' in question.lower():
                input_data = np.random.randn(1, 10)
                prediction = self.neural_network.forward_pass(input_data)
                return f"Neural network prediction: {prediction[0][0]:.4f}"
            
            else:
                return "I can discuss neural networks, machine learning algorithms, training processes, and predictions. What specific aspect interests you?"
        
        except Exception as e:
            return f"Error processing ML question: {str(e)}"
    
    def _process_web_search_question(self, question: str):
        """Process web search questions"""
        try:
            # Extract search query
            search_query = question.replace('search', '').replace('find', '').replace('look up', '').strip()
            if not search_query:
                search_query = question
            
            results = self.web_search.search(search_query, max_results=3)
            
            if results:
                response = f"I found {len(results)} results for '{search_query}':\n\n"
                for i, result in enumerate(results, 1):
                    response += f"{i}. {result['title']}\n"
                    if result['snippet']:
                        response += f"   {result['snippet'][:100]}...\n"
                    if result['url']:
                        response += f"   URL: {result['url']}\n"
                    response += "\n"
                return response
            else:
                return f"I couldn't find web results for '{search_query}'. This might be due to network issues or API limitations."
        
        except Exception as e:
            return f"Error performing web search: {str(e)}"
    
    def _generate_general_response(self, question: str):
        """Generate general AI response"""
        responses = [
            "That's an interesting question. I'm an AI assistant trained in discrete mathematics, machine learning, and neural networks. Could you be more specific about what you'd like to know?",
            "I'm here to help with AI, mathematics, logic, and various other topics. Can you provide more details about your question?",
            "I'd be happy to assist you. I can help with mathematical calculations, logical operations, machine learning concepts, and web searches. What specifically would you like to explore?",
            "Based on my training in discrete mathematics, neural networks, and machine learning, I can help with various topics. Could you clarify what information you're looking for?"
        ]
        return random.choice(responses)

class TerminalInterface:
    """Terminal/console interface for the AI assistant"""
    
    def __init__(self, ai_core: AICore, db_manager: DatabaseManager):
        self.ai_core = ai_core
        self.db_manager = db_manager
        self.running = False
        self.commands = {
            'help': self._show_help,
            'history': self._show_history,
            'clear': self._clear_screen,
            'settings': self._show_settings,
            'search': self._web_search,
            'exit': self._exit,
            'quit': self._exit
        }
    
    def start(self):
        """Start terminal interface"""
        self.running = True
        print("\n" + "="*50)
        print("AI Assistant Terminal Interface")
        print("="*50)
        print("Type 'help' for available commands")
        print("Type 'exit' or 'quit' to close")
        print("-"*50)
        
        while self.running:
            try:
                user_input = input("\nAI> ").strip()
                
                if not user_input:
                    continue
                
                # Check if it's a command
                if user_input.lower() in self.commands:
                    self.commands[user_input.lower()]()
                else:
                    # Process as AI question
                    response = self.ai_core.process_question(user_input)
                    print(f"\n[{response.source}] (Confidence: {response.confidence:.1%})")
                    print(f"Answer: {response.answer}")
                    
                    # Save to database
                    self.db_manager.save_conversation(response)
            
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def _show_help(self):
        """Show help information"""
        help_text = """
Available Commands:
- help: Show this help message
- history: Show conversation history
- clear: Clear the screen
- settings: Show current settings
- search <query>: Perform web search
- exit/quit: Exit the terminal

You can also ask any question directly, and I'll try to answer using:
- Discrete mathematics and logic
- Machine learning and neural networks
- Mathematical calculations
- Web search (if enabled)
- General AI knowledge
        """
        print(help_text)
    
    def _show_history(self):
        """Show conversation history"""
        history = self.db_manager.get_conversation_history(10)
        if history:
            print("\nRecent Conversation History:")
            print("-" * 40)
            for i, conv in enumerate(history, 1):
                print(f"{i}. Q: {conv.question[:50]}...")
                print(f"   A: {conv.answer[:100]}...")
                print(f"   [{conv.source}] {conv.timestamp.strftime('%Y-%m-%d %H:%M')}")
                print()
        else:
            print("No conversation history found.")
    
    def _clear_screen(self):
        """Clear the terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def _show_settings(self):
        """Show current settings"""
        api_key = self.db_manager.get_setting('api_key', 'Not set')
        search_engine = self.db_manager.get_setting('search_engine', 'duckduckgo')
        
        print(f"\nCurrent Settings:")
        print(f"- API Key: {'*' * len(api_key) if api_key != 'Not set' else 'Not set'}")
        print(f"- Search Engine: {search_engine}")
    
    def _web_search(self):
        """Perform web search"""
        query = input("Enter search query: ").strip()
        if query:
            response = self.ai_core.process_question(f"search {query}", use_web_search=True)
            print(f"\n{response.answer}")
        else:
            print("Please enter a search query.")
    
    def _exit(self):
        """Exit terminal"""
        self.running = False
        print("Goodbye!")

class MainApplication:
    """Main application class with GUI interface"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.ai_core = AICore()
        self.db_manager = DatabaseManager()
        self.terminal_interface = TerminalInterface(self.ai_core, self.db_manager)
        self.response_queue = queue.Queue()
        
        self.setup_gui()
        self.load_settings()
    
    def setup_gui(self):
        """Setup GUI interface"""
        self.root.title(f"Accurate AI Assistant v{VERSION}")
        self.root.geometry("1200x800")
        self.root.configure(bg=COLORS['white'])
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Red.TButton', background=COLORS['primary_red'], foreground=COLORS['white'])
        style.configure('Yellow.TLabel', background=COLORS['primary_yellow'], foreground=COLORS['black'])
        
        self.create_menu()
        self.create_main_interface()
        self.create_status_bar()
    
    def create_menu(self):
        """Create application menu"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Conversation", command=self.new_conversation)
        file_menu.add_command(label="Save Conversation", command=self.save_conversation)
        file_menu.add_command(label="Load Conversation", command=self.load_conversation)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Dashboard", command=self.show_dashboard)
        view_menu.add_command(label="History", command=self.show_history)
        view_menu.add_command(label="Neural Network Viz", command=self.show_neural_network)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Terminal", command=self.open_terminal)
        tools_menu.add_command(label="Web Search", command=self.open_web_search)
        tools_menu.add_command(label="Math Calculator", command=self.open_calculator)
        tools_menu.add_command(label="Logic Evaluator", command=self.open_logic_evaluator)
        
        # Settings menu
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        settings_menu.add_command(label="API Configuration", command=self.configure_api)
        settings_menu.add_command(label="Preferences", command=self.open_preferences)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="User Guide", command=self.show_user_guide)
    
    def create_main_interface(self):
        """Create main chat interface"""
        # Main frame
        main_frame = tk.Frame(self.root, bg=COLORS['white'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Welcome message
        welcome_frame = tk.Frame(main_frame, bg=COLORS['primary_yellow'], relief=tk.RAISED, bd=2)
        welcome_frame.pack(fill=tk.X, pady=(0, 10))
        
        welcome_label = tk.Label(welcome_frame, 
                                text="🤖 Welcome to Advanced AI Assistant! Ask me anything about AI, Mathematics, Logic, or any topic.",
                                bg=COLORS['primary_yellow'], fg=COLORS['black'], font=('Arial', 12, 'bold'),
                                wraplength=800)
        welcome_label.pack(pady=10)
        
        # Chat display area
        chat_frame = tk.Frame(main_frame)
        chat_frame.pack(fill=tk.BOTH, expand=True)
        
        self.chat_display = scrolledtext.ScrolledText(chat_frame, 
                                                     wrap=tk.WORD, 
                                                     height=20,
                                                     font=('Arial', 10),
                                                     bg=COLORS['white'],
                                                     fg=COLORS['black'])
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        
        # Configure text tags for styling
        self.chat_display.tag_configure("user", foreground=COLORS['primary_red'], font=('Arial', 10, 'bold'))
        self.chat_display.tag_configure("ai", foreground=COLORS['dark_red'], font=('Arial', 10))
        self.chat_display.tag_configure("system", foreground=COLORS['gray'], font=('Arial', 9, 'italic'))
        
        # Input frame
        input_frame = tk.Frame(main_frame, bg=COLORS['white'])
        input_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Question input
        self.question_var = tk.StringVar()
        self.question_entry = tk.Entry(input_frame, 
                                      textvariable=self.question_var,
                                      font=('Arial', 11),
                                      bg=COLORS['light_yellow'],
                                      fg=COLORS['black'],
                                      relief=tk.RAISED,
                                      bd=2)
        self.question_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.question_entry.bind('<Return>', self.process_question)
        
        # Control buttons
        button_frame = tk.Frame(input_frame, bg=COLORS['white'])
        button_frame.pack(side=tk.RIGHT)
        
        self.ask_button = tk.Button(button_frame, 
                                   text="Ask AI", 
                                   command=self.process_question,
                                   bg=COLORS['primary_red'], 
                                   fg=COLORS['white'],
                                   font=('Arial', 10, 'bold'),
                                   relief=tk.RAISED,
                                   bd=3)
        self.ask_button.pack(side=tk.LEFT, padx=2)
        
        search_button = tk.Button(button_frame, 
                                 text="Web Search", 
                                 command=self.toggle_web_search,
                                 bg=COLORS['primary_yellow'], 
                                 fg=COLORS['black'],
                                 font=('Arial', 9),
                                 relief=tk.RAISED,
                                 bd=2)
        search_button.pack(side=tk.LEFT, padx=2)
        
        clear_button = tk.Button(button_frame, 
                                text="Clear", 
                                command=self.clear_chat,
                                bg=COLORS['gray'], 
                                fg=COLORS['white'],
                                font=('Arial', 9),
                                relief=tk.RAISED,
                                bd=2)
        clear_button.pack(side=tk.LEFT, padx=2)
        
        # Options frame
        options_frame = tk.Frame(main_frame, bg=COLORS['white'])
        options_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.web_search_enabled = tk.BooleanVar()
        web_search_check = tk.Checkbutton(options_frame, 
                                         text="Enable Web Search", 
                                         variable=self.web_search_enabled,
                                         bg=COLORS['white'],
                                         fg=COLORS['black'],
                                         font=('Arial', 9))
        web_search_check.pack(side=tk.LEFT)
        
        self.confidence_label = tk.Label(options_frame, 
                                        text="Confidence: 0%", 
                                        bg=COLORS['white'],
                                        fg=COLORS['gray'],
                                        font=('Arial', 9))
        self.confidence_label.pack(side=tk.RIGHT)
    
    def create_status_bar(self):
        """Create status bar"""
        self.status_bar = tk.Label(self.root, 
                                  text="Ready", 
                                  relief=tk.SUNKEN, 
                                  anchor=tk.W,
                                  bg=COLORS['light_gray'],
                                  fg=COLORS['black'])
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def update_status(self, message):
        """Update status bar"""
        self.status_bar.config(text=f"{datetime.now().strftime('%H:%M:%S')} - {message}")
        self.root.update_idletasks()
    
    def process_question(self, event=None):
        """Process user question"""
        question = self.question_var.get().strip()
        if not question:
            return
        
        # Clear input
        self.question_var.set("")
        
        # Display user question
        self.chat_display.insert(tk.END, f"You: {question}\n", "user")
        self.chat_display.see(tk.END)
        
        # Update status
        self.update_status("Processing question...")
        
        # Disable ask button
        self.ask_button.config(state=tk.DISABLED)
        
        # Process in separate thread
        thread = threading.Thread(target=self._process_question_thread, args=(question,))
        thread.daemon = True
        thread.start()
        
        # Start checking for response
        self.root.after(100, self.check_response_queue)
    
    def _process_question_thread(self, question):
        """Process question in separate thread"""
        try:
            response = self.ai_core.process_question(question, self.web_search_enabled.get())
            self.response_queue.put(('success', response))
        except Exception as e:
            self.response_queue.put(('error', str(e)))
    
    def check_response_queue(self):
        """Check for AI response"""
        try:
            result_type, result = self.response_queue.get_nowait()
            
            if result_type == 'success':
                # Display AI response
                self.chat_display.insert(tk.END, f"AI ({result.source}): {result.answer}\n\n", "ai")
                self.chat_display.see(tk.END)
                
                # Update confidence
                self.confidence_label.config(text=f"Confidence: {result.confidence:.1%}")
                
                # Save to database
                self.db_manager.save_conversation(result)
                
                # Update status
                self.update_status("Response generated successfully")
            
            elif result_type == 'error':
                self.chat_display.insert(tk.END, f"Error: {result}\n\n", "system")
                self.chat_display.see(tk.END)
                self.update_status("Error processing question")
            
            # Re-enable ask button
            self.ask_button.config(state=tk.NORMAL)
            
        except queue.Empty:
            # Continue checking
            self.root.after(100, self.check_response_queue)
    
    def toggle_web_search(self):
        """Toggle web search option"""
        current = self.web_search_enabled.get()
        self.web_search_enabled.set(not current)
        status = "enabled" if not current else "disabled"
        self.update_status(f"Web search {status}")
    
    def clear_chat(self):
        """Clear chat display"""
        self.chat_display.delete(1.0, tk.END)
        self.confidence_label.config(text="Confidence: 0%")
        self.update_status("Chat cleared")
    
    def new_conversation(self):
        """Start new conversation"""
        self.clear_chat()
        self.chat_display.insert(tk.END, "Started new conversation. How can I help you?\n\n", "system")
    
    def save_conversation(self):
        """Save current conversation to file"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.chat_display.get(1.0, tk.END))
                self.update_status(f"Conversation saved to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save conversation: {e}")
    
    def load_conversation(self):
        """Load conversation from file"""
        try:
            filename = filedialog.askopenfilename(
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            if filename:
                with open(filename, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.chat_display.delete(1.0, tk.END)
                self.chat_display.insert(1.0, content)
                self.update_status(f"Conversation loaded from {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load conversation: {e}")
    
    def show_dashboard(self):
        """Show dashboard window"""
        dashboard = tk.Toplevel(self.root)
        dashboard.title("AI Assistant Dashboard")
        dashboard.geometry("800x600")
        dashboard.configure(bg=COLORS['white'])
        
        # Dashboard content
        notebook = ttk.Notebook(dashboard)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Statistics tab
        stats_frame = tk.Frame(notebook, bg=COLORS['white'])
        notebook.add(stats_frame, text="Statistics")
        
        history = self.db_manager.get_conversation_history(100)
        
        stats_text = scrolledtext.ScrolledText(stats_frame, wrap=tk.WORD, height=20)
        stats_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        stats_info = f"""AI Assistant Dashboard
{"-" * 50}
Total Conversations: {len(history)}
Database File: {DATABASE_FILE}
Configuration File: {CONFIG_FILE}
Application Version: {VERSION}

Recent Activity:
"""
        
        for i, conv in enumerate(history[:10], 1):
            stats_info += f"{i}. {conv.question[:50]}... [{conv.source}]\n"
        
        stats_text.insert(1.0, stats_info)
        
        # Neural Network tab
        nn_frame = tk.Frame(notebook, bg=COLORS['white'])
        notebook.add(nn_frame, text="Neural Network")
        
        self.create_neural_network_visualization(nn_frame)
    
    def create_neural_network_visualization(self, parent):
        """Create neural network visualization"""
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            fig.patch.set_facecolor(COLORS['white'])
            
            # Simple neural network diagram
            layers = [4, 6, 4, 2]
            layer_positions = []
            
            for i, layer_size in enumerate(layers):
                x = i * 2
                y_positions = np.linspace(-layer_size/2, layer_size/2, layer_size)
                layer_positions.append([(x, y) for y in y_positions])
                
                # Draw neurons
                for y in y_positions:
                    circle = plt.Circle((x, y), 0.2, color=COLORS['primary_red'], alpha=0.7)
                    ax.add_patch(circle)
            
            # Draw connections
            for i in range(len(layers) - 1):
                for pos1 in layer_positions[i]:
                    for pos2 in layer_positions[i + 1]:
                        ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
                               color=COLORS['gray'], alpha=0.3, linewidth=0.5)
            
            ax.set_xlim(-0.5, (len(layers) - 1) * 2 + 0.5)
            ax.set_ylim(-max(layers)/2 - 1, max(layers)/2 + 1)
            ax.set_title("Neural Network Architecture", fontsize=14, color=COLORS['dark_red'])
            ax.axis('off')
            
            canvas = FigureCanvasTkAgg(fig, parent)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            error_label = tk.Label(parent, text=f"Visualization error: {e}", 
                                 bg=COLORS['white'], fg=COLORS['primary_red'])
            error_label.pack(expand=True)
    
    def show_history(self):
        """Show conversation history window"""
        history_window = tk.Toplevel(self.root)
        history_window.title("Conversation History")
        history_window.geometry("700x500")
        history_window.configure(bg=COLORS['white'])
        
        # History display
        history_text = scrolledtext.ScrolledText(history_window, wrap=tk.WORD)
        history_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        history = self.db_manager.get_conversation_history(50)
        
        for i, conv in enumerate(history, 1):
            history_text.insert(tk.END, f"{i}. [{conv.timestamp.strftime('%Y-%m-%d %H:%M')}] "
                                       f"[{conv.source}] (Confidence: {conv.confidence:.1%})\n")
            history_text.insert(tk.END, f"Q: {conv.question}\n")
            history_text.insert(tk.END, f"A: {conv.answer}\n\n")
            history_text.insert(tk.END, "-" * 80 + "\n\n")
    
    def show_neural_network(self):
        """Show neural network visualization window"""
        nn_window = tk.Toplevel(self.root)
        nn_window.title("Neural Network Visualization")
        nn_window.geometry("800x600")
        nn_window.configure(bg=COLORS['white'])
        
        self.create_neural_network_visualization(nn_window)
    
    def open_terminal(self):
        """Open terminal interface in new thread"""
        terminal_thread = threading.Thread(target=self.terminal_interface.start)
        terminal_thread.daemon = True
        terminal_thread.start()
        self.update_status("Terminal interface opened")
    
    def open_web_search(self):
        """Open web search dialog"""
        query = simpledialog.askstring("Web Search", "Enter search query:")
        if query:
            self.question_var.set(f"search {query}")
            self.web_search_enabled.set(True)
            self.process_question()
    
    def open_calculator(self):
        """Open calculator dialog"""
        calc_window = tk.Toplevel(self.root)
        calc_window.title("Mathematical Calculator")
        calc_window.geometry("400x300")
        calc_window.configure(bg=COLORS['white'])
        
        # Calculator interface
        calc_frame = tk.Frame(calc_window, bg=COLORS['white'])
        calc_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        tk.Label(calc_frame, text="Mathematical Expression:", 
                bg=COLORS['white'], fg=COLORS['black'], font=('Arial', 12)).pack()
        
        calc_var = tk.StringVar()
        calc_entry = tk.Entry(calc_frame, textvariable=calc_var, font=('Arial', 14))
        calc_entry.pack(fill=tk.X, pady=10)
        
        result_label = tk.Label(calc_frame, text="Result will appear here", 
                               bg=COLORS['light_yellow'], fg=COLORS['black'], 
                               font=('Arial', 12), relief=tk.SUNKEN, bd=2)
        result_label.pack(fill=tk.X, pady=10)
        
        def calculate():
            expression = calc_var.get()
            if expression:
                response = self.ai_core.process_question(f"calculate {expression}")
                result_label.config(text=response.answer)
        
        calc_button = tk.Button(calc_frame, text="Calculate", command=calculate,
                               bg=COLORS['primary_red'], fg=COLORS['white'],
                               font=('Arial', 12, 'bold'))
        calc_button.pack(pady=10)
    
    def open_logic_evaluator(self):
        """Open logic evaluator dialog"""
        logic_window = tk.Toplevel(self.root)
        logic_window.title("Logic Expression Evaluator")
        logic_window.geometry("500x400")
        logic_window.configure(bg=COLORS['white'])
        
        # Logic evaluator interface
        logic_frame = tk.Frame(logic_window, bg=COLORS['white'])
        logic_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        tk.Label(logic_frame, text="Logical Expression:", 
                bg=COLORS['white'], fg=COLORS['black'], font=('Arial', 12)).pack()
        
        logic_var = tk.StringVar()
        logic_entry = tk.Entry(logic_frame, textvariable=logic_var, font=('Arial', 14))
        logic_entry.pack(fill=tk.X, pady=10)
        
        tk.Label(logic_frame, text="Examples: 'True AND False', 'A OR B', 'NOT C'", 
                bg=COLORS['white'], fg=COLORS['gray'], font=('Arial', 10)).pack()
        
        result_text = scrolledtext.ScrolledText(logic_frame, height=15, wrap=tk.WORD)
        result_text.pack(fill=tk.BOTH, expand=True, pady=10)
        
        def evaluate_logic():
            expression = logic_var.get()
            if expression:
                response = self.ai_core.process_question(expression)
                result_text.delete(1.0, tk.END)
                result_text.insert(1.0, f"Expression: {expression}\n\n{response.answer}")
        
        eval_button = tk.Button(logic_frame, text="Evaluate", command=evaluate_logic,
                               bg=COLORS['primary_yellow'], fg=COLORS['black'],
                               font=('Arial', 12, 'bold'))
        eval_button.pack(pady=5)
    
    def configure_api(self):
        """Configure API settings"""
        api_window = tk.Toplevel(self.root)
        api_window.title("API Configuration")
        api_window.geometry("500x300")
        api_window.configure(bg=COLORS['white'])
        
        # API configuration interface
        api_frame = tk.Frame(api_window, bg=COLORS['white'])
        api_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        tk.Label(api_frame, text="Search Engine API Configuration", 
                bg=COLORS['white'], fg=COLORS['black'], font=('Arial', 14, 'bold')).pack(pady=10)
        
        # API Key input
        tk.Label(api_frame, text="API Key:", 
                bg=COLORS['white'], fg=COLORS['black'], font=('Arial', 12)).pack(anchor=tk.W)
        
        api_key_var = tk.StringVar()
        current_key = self.db_manager.get_setting('api_key', '')
        api_key_var.set(current_key)
        
        api_key_entry = tk.Entry(api_frame, textvariable=api_key_var, show='*', font=('Arial', 12))
        api_key_entry.pack(fill=tk.X, pady=5)
        
        # Search Engine selection
        tk.Label(api_frame, text="Search Engine:", 
                bg=COLORS['white'], fg=COLORS['black'], font=('Arial', 12)).pack(anchor=tk.W, pady=(20, 5))
        
        search_engine_var = tk.StringVar()
        current_engine = self.db_manager.get_setting('search_engine', 'duckduckgo')
        search_engine_var.set(current_engine)
        
        engines = ['duckduckgo', 'google', 'bing']
        for engine in engines:
            tk.Radiobutton(api_frame, text=engine.title(), variable=search_engine_var, 
                          value=engine, bg=COLORS['white'], fg=COLORS['black']).pack(anchor=tk.W)
        
        def save_api_config():
            api_key = api_key_var.get()
            search_engine = search_engine_var.get()
            
            self.db_manager.save_setting('api_key', api_key)
            self.db_manager.save_setting('search_engine', search_engine)
            
            # Update AI core with new settings
            self.ai_core.web_search = WebSearchEngine(api_key)
            
            messagebox.showinfo("Success", "API configuration saved successfully!")
            api_window.destroy()
        
        save_button = tk.Button(api_frame, text="Save Configuration", command=save_api_config,
                               bg=COLORS['primary_red'], fg=COLORS['white'],
                               font=('Arial', 12, 'bold'))
        save_button.pack(pady=20)
    
    def open_preferences(self):
        """Open preferences dialog"""
        pref_window = tk.Toplevel(self.root)
        pref_window.title("Preferences")
        pref_window.geometry("400x300")
        pref_window.configure(bg=COLORS['white'])
        
        # Preferences interface
        pref_frame = tk.Frame(pref_window, bg=COLORS['white'])
        pref_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        tk.Label(pref_frame, text="Application Preferences", 
                bg=COLORS['white'], fg=COLORS['black'], font=('Arial', 14, 'bold')).pack(pady=10)
        
        # Auto-save conversations
        auto_save_var = tk.BooleanVar()
        auto_save_check = tk.Checkbutton(pref_frame, text="Auto-save conversations", 
                                        variable=auto_save_var, bg=COLORS['white'], fg=COLORS['black'])
        auto_save_check.pack(anchor=tk.W, pady=5)
        
        # Show confidence scores
        show_confidence_var = tk.BooleanVar()
        confidence_check = tk.Checkbutton(pref_frame, text="Show confidence scores", 
                                         variable=show_confidence_var, bg=COLORS['white'], fg=COLORS['black'])
        confidence_check.pack(anchor=tk.W, pady=5)
        
        # Enable web search by default
        default_search_var = tk.BooleanVar()
        search_check = tk.Checkbutton(pref_frame, text="Enable web search by default", 
                                     variable=default_search_var, bg=COLORS['white'], fg=COLORS['black'])
        search_check.pack(anchor=tk.W, pady=5)
        
        def save_preferences():
            # Save preferences to database
            self.db_manager.save_setting('auto_save', str(auto_save_var.get()))
            self.db_manager.save_setting('show_confidence', str(show_confidence_var.get()))
            self.db_manager.save_setting('default_search', str(default_search_var.get()))
            
            messagebox.showinfo("Success", "Preferences saved successfully!")
            pref_window.destroy()
        
        save_button = tk.Button(pref_frame, text="Save Preferences", command=save_preferences,
                               bg=COLORS['primary_yellow'], fg=COLORS['black'],
                               font=('Arial', 12, 'bold'))
        save_button.pack(pady=20)
    
    def show_about(self):
        """Show about dialog"""
        about_text = f""" Accurate AI Assistant v{VERSION}

Author:Ian Carter Kulani
E-mail:iancarterkulani@gmail.com
Phone:+265(0)988061969

A comprehensive AI application featuring:
• Discrete Mathematics Processing
• Machine Learning Algorithms
• Neural Network Simulation
• Web Search Integration
• Terminal Interface
• Graphical User Interface

Built with Python, Tkinter, NumPy, Matplotlib
Developed with modern AI principles

© 2024 AI Assistant Project
        """
        messagebox.showinfo("About AI Assistant", about_text)
    
    def show_user_guide(self):
        """Show user guide"""
        guide_window = tk.Toplevel(self.root)
        guide_window.title("User Guide")
        guide_window.geometry("700x600")
        guide_window.configure(bg=COLORS['white'])
        
        guide_text = scrolledtext.ScrolledText(guide_window, wrap=tk.WORD)
        guide_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        user_guide = """AI ASSISTANT USER GUIDE
======================

GETTING STARTED
--------------
1. Type your question in the input field at the bottom
2. Click "Ask AI" or press Enter to get a response
3. Enable "Web Search" for internet-based queries
4. Use the menu bar for advanced features

QUESTION TYPES
--------------
• General Questions: Ask anything and get AI responses
• Mathematical: "Calculate 15 + 27" or "What is 5 * 8?"
• Logical: "Evaluate True AND False" or "Generate truth table for A, B"
• Machine Learning: "Explain neural networks" or "What is training?"
• Web Search: "Search for Python tutorials" (requires web search enabled)

MENU FEATURES
------------
FILE MENU:
- New Conversation: Start fresh chat
- Save/Load Conversation: Export/import chat history
- Exit: Close application

VIEW MENU:
- Dashboard: View statistics and neural network visualization
- History: Browse previous conversations
- Neural Network Viz: See network architecture

TOOLS MENU:
- Terminal: Open command-line interface
- Web Search: Direct web search dialog
- Math Calculator: Dedicated calculation tool
- Logic Evaluator: Evaluate logical expressions

SETTINGS MENU:
- API Configuration: Set up search engine API keys
- Preferences: Customize application behavior

TERMINAL INTERFACE
-----------------
Commands available in terminal mode:
- help: Show available commands
- history: View conversation history
- clear: Clear terminal screen
- settings: Show current settings
- search <query>: Perform web search
- exit/quit: Close terminal

API CONFIGURATION
----------------
1. Go to Settings > API Configuration
2. Enter your search engine API key
3. Select preferred search engine (DuckDuckGo, Google, Bing)
4. Save configuration

Note: DuckDuckGo works without API key (limited results)
Google and Bing require valid API keys for full functionality

TIPS FOR BEST RESULTS
---------------------
• Be specific in your questions
• Use mathematical expressions for calculations
• Enable web search for current information
• Save important conversations using File > Save
• Check the confidence score for AI responses
• Use the terminal for batch operations

TROUBLESHOOTING
--------------
• If web search fails, check your internet connection
• For API errors, verify your API key in settings
• Clear chat if display becomes cluttered
• Restart application if performance degrades
• Check the status bar for current operation status

KEYBOARD SHORTCUTS
-----------------
• Enter: Send question/command
• Ctrl+N: New conversation
• Ctrl+S: Save conversation
• Ctrl+O: Open conversation
• Ctrl+Q: Quit application

For additional help, visit the Help menu or contact support.
        """
        
        guide_text.insert(1.0, user_guide)
        guide_text.config(state=tk.DISABLED)
    
    def load_settings(self):
        """Load application settings"""
        try:
            # Load web search default setting
            default_search = self.db_manager.get_setting('default_search', 'False')
            self.web_search_enabled.set(default_search.lower() == 'true')
            
            # Load API key for web search
            api_key = self.db_manager.get_setting('api_key')
            if api_key:
                self.ai_core.web_search = WebSearchEngine(api_key)
            
            self.update_status("Settings loaded successfully")
        except Exception as e:
            logger.error(f"Error loading settings: {e}")
            self.update_status("Using default settings")
    
    def run(self):
        """Run the application"""
        self.update_status("AI Assistant ready - Ask me anything!")
        self.root.mainloop()

def main():
    """Main function to run the AI Assistant"""
    try:
        # Check dependencies
        required_modules = ['tkinter', 'numpy', 'matplotlib', 'requests', 'sqlite3']
        missing_modules = []
        
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing_modules.append(module)
        
        if missing_modules:
            print(f"Missing required modules: {', '.join(missing_modules)}")
            print("Please install them using: pip install " + ' '.join(missing_modules))
            return
        
        # Create and run application
        app = MainApplication()
        app.run()
        
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        print(f"Application error: {e}")
        logger.error(f"Application error: {e}", exc_info=True)

if __name__ == "__main__":
    main()