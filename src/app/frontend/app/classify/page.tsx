import Link from 'next/link';
import { ArrowRight, Github, Linkedin, BarChart3 } from 'lucide-react';

export default function Homepage() {
  return (
    <div>
      <main className="container mx-auto px-4 py-16">
        <h2 className="text-3xl text-center font-bold mb-4">Classify Your Image</h2>
        <p className="mb-8">Upload a histopathology image to see the classification results.</p>
        <form className="flex flex-col gap-4">
          <input type="file" accept="image/*" className="border border-gray-300 p-2 rounded" />
          <button type="button" className="bg-gray-200 text-xl px-4 py-2 rounded">Use Sample Image</button>
          <button type="submit" className="bg-blue-600 text-xl text-white px-4 py-2 rounded">Classify!</button>
        </form>
      </main>
    </div>
  );
}