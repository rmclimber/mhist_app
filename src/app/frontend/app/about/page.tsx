import Link from 'next/link';
import { ArrowRight, Github, Linkedin, BarChart3 } from 'lucide-react';

export default function Homepage() {
  return (
    <div>
      <main className="container mx-auto px-4 py-16">
        <div className="max-w-4xl mx-auto text-center space-y-8">
          <h3 className="text-4xl font-bold text-gray-900 tracking-tight">MHIST Classifier Overview</h3>
        </div>
        <div className="mt-8 prose prose-lg prose-blue mx-auto text-left">
          <p>
            The MHIST Classifier app is an end-to-end demonstration of machine
            learning for histopathology image classification using the MHIST 
            dataset. Feel free to upload your own images to see how the model
            performs, or use a randomly-selected test set example provided by 
            the app.
          </p>
        <div className="text-3xl font-bold text-center mt-6">Technical details:</div>
        <p>
          The <a href="https://huggingface.co/google/vit-base-patch16-224"
                target="_blank" 
                rel="noopener noreferrer" 
                title="Hugging Face model card"
                className="text-blue-600 hover:text-blue-700 transition-colors">core 
                model</a> is a Vision Transformer (ViT) configured for images
                sized at 224x224, using 16x16 patches, and pre-trained on 
                ImageNet. The model was fine-tuned on the MHIST dataset.

        </p>

        <p>
          The model was trained using PyTorch Lightning via Vertex AI on
          Google Cloud Platform, with experiment-tracking via Weights & Biases.
        </p>

        <p>
          The model is served via a FastAPI backend, with endpoints for image
          upload and classification. The frontend is built with Next.js and 
          Tailwind CSS for styling. The entire application is containerized
          using Docker and deployed on Google Cloud Platform.
        </p>
          <Link href="/">Return to homepage</Link>
        </div>
      </main>
    </div>
  );
}