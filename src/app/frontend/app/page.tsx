import Link from 'next/link';
import { ArrowRight, Github, Linkedin, BarChart3 } from 'lucide-react';

export default function Homepage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-300 to-purple-300">
      {/* Header Section */}
      <header className="border-b bg-white/50 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <h1 className="text-xl font-bold text-gray-900">MHIST Classifier</h1>
          </div>
        
        <div>
          <p>Rick Morris</p>
        </div>
      
        <nav className="flex gap-6">
          <a 
            href={process.env.NEXT_PUBLIC_GITHUB_URL || '#'} 
            target="_blank" 
            rel="noopener noreferrer" 
            title="GitHub Repository"
            aria-label="GitHub Repository"
            className="flex items-center gap-1 text-gray-700 hover:text-gray-900">
            <Github className="w-5 h-5" />
          </a>
          <a 
            href={process.env.NEXT_PUBLIC_LINKEDIN_URL || '#'} 
            target="_blank" 
            rel="noopener noreferrer" 
            title="LinkedIn Profile"
            aria-label="LinkedIn Profile"
            className="flex items-center gap-1 text-gray-700 hover:text-gray-900">
            <Linkedin className="w-5 h-5" />
          </a>
          <a 
            href={process.env.NEXT_PUBLIC_WANDB_URL || '#'} 
            target="_blank" 
            rel="noopener noreferrer"
            title="Weights & Biases"
            aria-label="Weights & Biases"
            className="text-gray-600 hover:text-gray-900 transition-colors"
          >
            <BarChart3 className="w-5 h-5" />
          </a>
        </nav>
        </div>
      </header>

    </div>
  );
}   