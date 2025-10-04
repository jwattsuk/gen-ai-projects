"""
Testing and Evaluation Framework
Provides comprehensive testing and evaluation metrics for the RAG chatbot
"""

import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.metrics import average_precision_score
from chatbot import RAGChatBot
from datetime import datetime


class TestCase:
    """Represents a single test case for the chatbot"""
    
    def __init__(self, 
                 query: str, 
                 expected_keywords: List[str] = None,
                 relevant_responses: List[str] = None,
                 irrelevant_responses: List[str] = None,
                 expected_sources: List[str] = None):
        """
        Initialize test case
        
        Args:
            query: The test query
            expected_keywords: Keywords that should appear in the response
            relevant_responses: Examples of relevant responses
            irrelevant_responses: Examples of irrelevant responses
            expected_sources: Expected source files that should be retrieved
        """
        self.query = query
        self.expected_keywords = expected_keywords or []
        self.relevant_responses = relevant_responses or []
        self.irrelevant_responses = irrelevant_responses or []
        self.expected_sources = expected_sources or []


class ChatBotEvaluator:
    """Evaluates chatbot performance using various metrics"""
    
    def __init__(self, chatbot: RAGChatBot):
        """
        Initialize evaluator
        
        Args:
            chatbot: RAGChatBot instance to evaluate
        """
        self.chatbot = chatbot
        self.test_results = []
    
    def evaluate_keyword_presence(self, response: str, keywords: List[str]) -> float:
        """
        Evaluate how many expected keywords are present in the response
        
        Args:
            response: Generated response
            keywords: Expected keywords
            
        Returns:
            Keyword presence score (0-1)
        """
        if not keywords:
            return 1.0
        
        response_lower = response.lower()
        present_keywords = sum(1 for keyword in keywords if keyword.lower() in response_lower)
        return present_keywords / len(keywords)
    
    def evaluate_source_relevance(self, sources: List[Dict[str, Any]], expected_sources: List[str]) -> float:
        """
        Evaluate if the retrieved sources are relevant
        
        Args:
            sources: Retrieved sources from chatbot
            expected_sources: Expected source files
            
        Returns:
            Source relevance score (0-1)
        """
        if not expected_sources:
            return 1.0
        
        if not sources:
            return 0.0
        
        retrieved_paths = [source['path'] for source in sources]
        relevant_count = 0
        
        for expected_source in expected_sources:
            for retrieved_path in retrieved_paths:
                if expected_source in retrieved_path:
                    relevant_count += 1
                    break
        
        return relevant_count / len(expected_sources)
    
    def evaluate_response_quality(self, response: str, relevant_responses: List[str], irrelevant_responses: List[str]) -> float:
        """
        Evaluate response quality using semantic similarity (simplified)
        
        Args:
            response: Generated response
            relevant_responses: Examples of good responses
            irrelevant_responses: Examples of bad responses
            
        Returns:
            Quality score (0-1)
        """
        if not relevant_responses and not irrelevant_responses:
            return 0.5  # Neutral score when no examples provided
        
        response_lower = response.lower()
        
        # Simple keyword-based similarity
        relevant_score = 0
        if relevant_responses:
            for relevant_resp in relevant_responses:
                # Count overlapping words
                response_words = set(response_lower.split())
                relevant_words = set(relevant_resp.lower().split())
                overlap = len(response_words.intersection(relevant_words))
                relevant_score = max(relevant_score, overlap / max(len(response_words), len(relevant_words)))
        
        irrelevant_penalty = 0
        if irrelevant_responses:
            for irrelevant_resp in irrelevant_responses:
                response_words = set(response_lower.split())
                irrelevant_words = set(irrelevant_resp.lower().split())
                overlap = len(response_words.intersection(irrelevant_words))
                irrelevant_penalty = max(irrelevant_penalty, overlap / max(len(response_words), len(irrelevant_words)))
        
        return max(0, relevant_score - irrelevant_penalty)
    
    def run_single_test(self, test_case: TestCase) -> Dict[str, Any]:
        """
        Run a single test case
        
        Args:
            test_case: TestCase to evaluate
            
        Returns:
            Test results dictionary
        """
        print(f"Testing query: {test_case.query}")
        
        # Get chatbot response
        result = self.chatbot.chat(test_case.query)
        response = result['response']
        sources = result['sources']
        
        # Evaluate different aspects
        keyword_score = self.evaluate_keyword_presence(response, test_case.expected_keywords)
        source_score = self.evaluate_source_relevance(sources, test_case.expected_sources)
        quality_score = self.evaluate_response_quality(
            response, test_case.relevant_responses, test_case.irrelevant_responses
        )
        
        # Calculate overall score
        overall_score = (keyword_score + source_score + quality_score) / 3
        
        test_result = {
            'query': test_case.query,
            'response': response,
            'sources': sources,
            'keyword_score': keyword_score,
            'source_score': source_score,
            'quality_score': quality_score,
            'overall_score': overall_score,
            'timestamp': datetime.now().isoformat()
        }
        
        self.test_results.append(test_result)
        return test_result
    
    def run_test_suite(self, test_cases: List[TestCase]) -> Dict[str, Any]:
        """
        Run a complete test suite
        
        Args:
            test_cases: List of TestCase objects
            
        Returns:
            Aggregated test results
        """
        print(f"Running test suite with {len(test_cases)} test cases...")
        
        all_results = []
        for i, test_case in enumerate(test_cases):
            print(f"\nTest {i+1}/{len(test_cases)}")
            result = self.run_single_test(test_case)
            all_results.append(result)
        
        # Calculate aggregate metrics
        avg_keyword_score = np.mean([r['keyword_score'] for r in all_results])
        avg_source_score = np.mean([r['source_score'] for r in all_results])
        avg_quality_score = np.mean([r['quality_score'] for r in all_results])
        avg_overall_score = np.mean([r['overall_score'] for r in all_results])
        
        suite_results = {
            'test_results': all_results,
            'summary': {
                'total_tests': len(test_cases),
                'avg_keyword_score': avg_keyword_score,
                'avg_source_score': avg_source_score,
                'avg_quality_score': avg_quality_score,
                'avg_overall_score': avg_overall_score,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        return suite_results
    
    def calculate_map_score(self, test_cases: List[TestCase]) -> float:
        """
        Calculate Mean Average Precision (MAP) score similar to the notebook
        
        Args:
            test_cases: List of test cases with relevant/irrelevant responses
            
        Returns:
            MAP score
        """
        total_average_precision = 0
        valid_tests = 0
        
        for test_case in test_cases:
            if not test_case.relevant_responses or not test_case.irrelevant_responses:
                continue
            
            # Get chatbot response
            result = self.chatbot.chat(test_case.query)
            response = result['response']
            
            # Create binary labels
            all_responses = test_case.relevant_responses + test_case.irrelevant_responses
            true_labels = [1] * len(test_case.relevant_responses) + [0] * len(test_case.irrelevant_responses)
            
            # Simple scoring: 1 if response matches, 0 otherwise
            predicted_scores = [1 if resp.lower() in response.lower() else 0 for resp in all_responses]
            
            # Calculate average precision if we have any positive predictions
            if sum(predicted_scores) > 0:
                try:
                    ap = average_precision_score(true_labels, predicted_scores)
                    total_average_precision += ap
                    valid_tests += 1
                except ValueError:
                    # Skip if all labels are the same
                    continue
        
        return total_average_precision / valid_tests if valid_tests > 0 else 0.0
    
    def save_results(self, filepath: str):
        """Save test results to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        print(f"Results saved to {filepath}")
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a human-readable test report"""
        summary = results['summary']
        
        report = f"""
RAG Chatbot Test Report
======================
Generated: {summary['timestamp']}

Summary Metrics:
- Total Tests: {summary['total_tests']}
- Average Keyword Score: {summary['avg_keyword_score']:.3f}
- Average Source Score: {summary['avg_source_score']:.3f}
- Average Quality Score: {summary['avg_quality_score']:.3f}
- Overall Average Score: {summary['avg_overall_score']:.3f}

Individual Test Results:
"""
        
        for i, result in enumerate(results['test_results']):
            report += f"""
Test {i+1}: {result['query']}
- Keywords: {result['keyword_score']:.3f}
- Sources: {result['source_score']:.3f}
- Quality: {result['quality_score']:.3f}
- Overall: {result['overall_score']:.3f}
"""
        
        return report


def create_default_test_cases() -> List[TestCase]:
    """Create a default set of test cases for AI/ML topics"""
    
    test_cases = [
        TestCase(
            query="What is a perceptron?",
            expected_keywords=["perceptron", "neural", "artificial", "neuron", "classifier"],
            relevant_responses=[
                "A perceptron is a type of artificial neuron.",
                "It's a binary classifier used in machine learning.",
                "A perceptron is a fundamental building block of neural networks."
            ],
            irrelevant_responses=[
                "A perceptron is a type of fruit.",
                "It's a type of car.",
                "Perceptrons are used for cooking."
            ],
            expected_sources=["perceptron.md"]
        ),
        
        TestCase(
            query="What is machine learning?",
            expected_keywords=["machine learning", "data", "algorithms", "patterns", "artificial intelligence"],
            relevant_responses=[
                "Machine learning is a method of data analysis that automates analytical model building.",
                "It's a branch of artificial intelligence based on the idea that systems can learn from data.",
                "Machine learning enables systems to learn and improve from experience."
            ],
            irrelevant_responses=[
                "Machine learning is a type of fruit.",
                "It's a type of car.",
                "Machine learning is about fixing machines."
            ],
            expected_sources=["frameworks.md", "own_framework.md"]
        ),
        
        TestCase(
            query="How do neural networks work?",
            expected_keywords=["neural networks", "neurons", "layers", "weights", "activation"],
            relevant_responses=[
                "Neural networks work by connecting artificial neurons in layers.",
                "They process information through interconnected nodes that mimic brain neurons.",
                "Neural networks learn by adjusting weights between connections."
            ],
            irrelevant_responses=[
                "Neural networks are fishing nets.",
                "They work by connecting computers with cables.",
                "Neural networks are social networks for nerds."
            ],
            expected_sources=["frameworks.md", "perceptron.md"]
        ),
        
        TestCase(
            query="What is deep learning?",
            expected_keywords=["deep learning", "neural networks", "layers", "artificial intelligence"],
            relevant_responses=[
                "Deep learning is a subset of machine learning using deep neural networks.",
                "It uses multiple layers to progressively extract features from raw input.",
                "Deep learning can automatically discover patterns in data."
            ],
            irrelevant_responses=[
                "Deep learning is learning while underwater.",
                "It's about digging deep holes.",
                "Deep learning is philosophical thinking."
            ],
            expected_sources=["frameworks.md", "own_framework.md"]
        )
    ]
    
    return test_cases


if __name__ == "__main__":
    # Example usage
    
    # Initialize chatbot
    chatbot = RAGChatBot()
    chatbot.load_knowledge_base("embeddings")
    
    # Create evaluator
    evaluator = ChatBotEvaluator(chatbot)
    
    # Create test cases
    test_cases = create_default_test_cases()
    
    # Run tests
    results = evaluator.run_test_suite(test_cases)
    
    # Calculate MAP score
    map_score = evaluator.calculate_map_score(test_cases)
    print(f"\nMean Average Precision (MAP) Score: {map_score:.3f}")
    
    # Generate and print report
    report = evaluator.generate_report(results)
    print(report)
    
    # Save results
    evaluator.save_results(f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    # Save report
    with open(f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", 'w') as f:
        f.write(report)
    
    print("Test completed! Results and report saved.")