import React, { useState } from "react";
import { motion } from "framer-motion";
import { Upload, Music, Type, SlidersHorizontal, Image as ImageIcon, Wand2 } from "lucide-react";

export default function App() {
  const [videoFile, setVideoFile] = useState(null);
  const [bgmFile, setBgmFile] = useState(null);
  const [captionText, setCaptionText] = useState("");
  const [textColor, setTextColor] = useState("#ffffff");
  const [font, setFont] = useState("Poppins");
  const [filters, setFilters] = useState({ brightness: 100, contrast: 100, saturate: 100 });
  const [overlayFile, setOverlayFile] = useState(null);
  const [logoFile, setLogoFile] = useState(null);

  return (
    <div className="min-h-screen w-full bg-gray-900 text-white p-6">
      <h1 className="text-4xl font-bold text-center mb-8">AI Video Editor</h1>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-5xl mx-auto">

        {/* VIDEO UPLOAD */}
        <motion.div className="p-6 bg-gray-800 rounded-2xl shadow-xl" whileHover={{ scale: 1.02 }}>
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2"><Upload /> Upload Video</h2>
          <input type="file" accept="video/*" onChange={(e) => setVideoFile(e.target.files[0])} className="w-full" />
        </motion.div>

        {/* BGM UPLOAD */}
        <motion.div className="p-6 bg-gray-800 rounded-2xl shadow-xl" whileHover={{ scale: 1.02 }}>
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2"><Music /> Background Music</h2>
          <input type="file" accept="audio/*" onChange={(e) => setBgmFile(e.target.files[0])} />
        </motion.div>

        {/* CAPTIONS */}
        <motion.div className="p-6 bg-gray-800 rounded-2xl shadow-xl col-span-1 md:col-span-2" whileHover={{ scale: 1.02 }}>
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2"><Type /> Captions</h2>
          <textarea value={captionText} onChange={(e) => setCaptionText(e.target.value)} className="w-full p-3 rounded-lg bg-gray-700"></textarea>

          <div className="flex gap-4 mt-4">
            <div>
              <label className="block mb-1">Text Color</label>
              <input type="color" value={textColor} onChange={(e) => setTextColor(e.target.value)} />
            </div>
            <div>
              <label className="block mb-1">Font</label>
              <select value={font} onChange={(e) => setFont(e.target.value)} className="bg-gray-700 p-2 rounded-lg">
                <option>Poppins</option>
                <option>Montserrat</option>
                <option>Roboto</option>
                <option>Oswald</option>
                <option>Lobster</option>
              </select>
            </div>
          </div>
        </motion.div>

        {/* FILTERS */}
        <motion.div className="p-6 bg-gray-800 rounded-2xl shadow-xl" whileHover={{ scale: 1.02 }}>
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2"><SlidersHorizontal /> Filters</h2>

          {Object.keys(filters).map((item) => (
            <div key={item} className="mb-3">
              <label className="block mb-1 capitalize">{item}</label>
              <input type="range" min="50" max="150" value={filters[item]} onChange={(e) => setFilters({ ...filters, [item]: e.target.value })} className="w-full" />
            </div>
          ))}
        </motion.div>

        {/* OVERLAY */}
        <motion.div className="p-6 bg-gray-800 rounded-2xl shadow-xl" whileHover={{ scale: 1.02 }}>
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2"><ImageIcon /> Overlay</h2>
          <input type="file" accept="image/*" onChange={(e) => setOverlayFile(e.target.files[0])} />
        </motion.div>

        {/* LOGO UPLOAD */}
        <motion.div className="p-6 bg-gray-800 rounded-2xl shadow-xl" whileHover={{ scale: 1.02 }}>
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2"><Wand2 /> Logo</h2>
          <input type="file" accept="image/*" onChange={(e) => setLogoFile(e.target.files[0])} />
        </motion.div>

      </div>

      {/* FINAL BUTTON */}
      <div className="text-center mt-10">
        <button className="px-6 py-3 text-lg bg-blue-600 rounded-xl shadow-lg hover:bg-blue-700">Generate Final Video</button>
      </div>
    </div>
  );
}
