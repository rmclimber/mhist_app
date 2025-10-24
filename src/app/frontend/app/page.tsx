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
      {/* Content Section */}
      <main className="container mx-auto px-4 py-16">
        <div className="max-w-4xl mx-auto text-center space-y-8">
          {/* Title */}
          <div className="space-y-4">
            <h2 className="text-5xl font-bold text-gray-900 tracking-tight">
              MHIST Image Classification <span className="block text-blue-600 mt-2">Demo</span>
            </h2>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              Upload histopathology images and see machine learning in action!
            </p>
          </div>
          {/*CTA Buttons */}
          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
            <Link href="/classify" className="group inline-flex items-center gap-2 bg-blue-600 text-white px-8 py-3 rounded-lg font semibold hover:bg-blue-700 transition-all shadow-lg hover:shadow-xl">
              Try Classifier <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </Link>
            <Link href="/about" className="infline-flex items-center gap-2 bg-white text-gray-700 px-8 py-3 rounded-lg font semibold hover:bg-gray-50 transition-colors border border-gray-200">
              Learn More
            </Link>
          </div>
          {/* Project links */}
          <div className="mt-16 p-6 bg-white rounded-lg border border-gray-200 shadow-sm">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Project links</h3>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <a href={process.env.NEXT_PUBLIC_GITHUB_URL || '#'} 
                target="_blank" rel="noopener noreferrer"
                className="inline-flex items-center gap-2 text-gray-700 hover:text-gray-900 transition-colors">
                  <Github className="w-5 h-5">View on Github</Github>
                </a>
                <span className="hidden sm:block text-gray-300">|</span>
                <a href={process.env.NEXT_PUBLIC_WANDB_URL || '#'} 
                  target="_blank" rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 text-gray-700 hover:text-gray-900 transition-colors">
                    <BarChart3 className="w-5 h-5">Weights & Biases Experiment</BarChart3>
                </a>
            </div>
          </div>
        </div>
      </main>

    </div>
  );
}   