'use client';

import { useState } from 'react';

export default function ClassifyPage() {
    const [ imgFile, setImgFile ] = useState<File | null>(null);
    const [ previewUrl, setPreviewUrl ] = useState<string | null>(null);

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0] || null;
      setImgFile(file);
      const url = file ? URL.createObjectURL(file) : null;
      setPreviewUrl(url);
    };

  return (
    <div>
      <main className="container mx-auto px-4 py-16">
        <h2 className="text-3xl text-center font-bold mb-4">Classify Your Image</h2>
        <div className="max-w-3xl mx-auto">
            <p className="mb-8">Upload a histopathology image to see the classification results.</p>
            
            <form className="flex flex-col gap-4">
                <input
                    onChange={handleFileChange}
                    type="file"
                    accept="image/*"
                    className="border border-gray-300 p-2 rounded"
                />
                {/* Image preview */}
                <div className="text-center flex justify-center">
                    {previewUrl ? (
                        <div>
                            <img src={previewUrl} alt="Image Preview" className="mt-4 max-w-xs border rounded items-center" />
                    <p className="mt-4 text-gray-500">{imgFile.name}</p>
                    </div>
                ) : (
                    <p className="mt-4 text-gray-500">No image selected</p>
                )}
                </div>

            <button type="button" className="bg-gray-200 text-xl px-4 py-2 rounded">Use Sample Image</button>
            <button type="submit" className="bg-blue-600 text-xl text-white px-4 py-2 rounded">Classify!</button>
            </form>
        </div>
      </main>
    </div>
  );
}