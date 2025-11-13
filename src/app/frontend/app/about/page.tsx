import Link from 'next/link';
import { ArrowRight, Github, Linkedin, BarChart3 } from 'lucide-react';

export default function Homepage() {
  return (
    <div>
      <main className="container mx-auto px-4 py-16">
        <div className="max-w-4xl mx-auto text-center space-y-8">
          <h1>About MHIST Classifier</h1>
          <p>
            The MHIST Classifier is a state-of-the-art tool for histopathology image analysis.
          </p>
          <Link href="/">Go back to homepage</Link>
        </div>
      </main>
    </div>
  );
}