'use client';

import { useState } from 'react';
import { Send, AlertCircle, CheckCircle, Clock } from 'lucide-react';
import LocationSelector from '@/components/LocationSelector';

interface ReviewData {
  name: string;
  category: string;
  description: string;
  review: string;
  rating: number;
}

interface LogEntry {
  type: string;
  message: string;
}

interface MapLocation {
  lat: number;
  lng: number;
  name?: string;
}

export default function ReviewAnalyzer() {
  const [reviewData, setReviewData] = useState<ReviewData>({
    name: '',
    category: '',
    description: '',
    review: '',
    rating: 3
  });
  
  const [location, setLocation] = useState<MapLocation | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchValue, setSearchValue] = useState('');

  const handleLocationChange = (newLocation: MapLocation | null, category: string, description: string) => {
    setLocation(newLocation);
    setReviewData(prev => ({ 
      ...prev, 
      name: newLocation?.name || '',
      category,
      description
    }));
  };

  const handleAnalyze = async () => {
    if (!reviewData.category || !reviewData.description || !reviewData.review || !location) {
      setError('Please fill in all fields and select a location');
      return;
    }

    setIsAnalyzing(true);
    setError(null);
    setLogs([]);

    try {
      const response = await fetch('http://127.0.0.1:8000/analyze_review/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          review: {
            ...reviewData,
            name: location.name || reviewData.name,
          }
        })
      });

      if (response.ok) {
        const data = await response.json();
        setLogs(data.logs || []);
      } else {
        setError(`Backend error: ${response.status} - ${await response.text()}`);
      }
    } catch (err) {
      setError(`Connection error: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const getLogIcon = (type: string) => {
    switch (type.toLowerCase()) {
      case 'success':
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'warning':
        return <AlertCircle className="w-4 h-4 text-yellow-500" />;
      default:
        return <Clock className="w-4 h-4 text-blue-500" />;
    }
  };

  const getLogStyle = (type: string) => {
    switch (type.toLowerCase()) {
      case 'success':
        return 'border-green-200 bg-green-50 text-green-800';
      case 'warning':
        return 'border-yellow-200 bg-yellow-50 text-yellow-800';
      default:
        return 'border-blue-200 bg-blue-50 text-blue-800';
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 p-6">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-white mb-2">ARC</h1>
          <p className="text-gray-300">Analyse location reviews using ML-powered detection</p>
        </div>

        <div className="bg-gray-800 rounded-xl shadow-lg p-6 mb-8">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Location Selection */}
            <LocationSelector
              location={location}
              onLocationChange={handleLocationChange}
              searchValue={searchValue}
              onSearchValueChange={setSearchValue}
            />

            {/* Form Fields */}
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-200 mb-1">Category</label>
                <input
                  type="text"
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="e.g., Restaurant, Hotel, Shop"
                  value={reviewData.category}
                  onChange={(e) => setReviewData(prev => ({ ...prev, category: e.target.value }))}
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-200 mb-1">Description</label>
                <textarea
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  rows={3}
                  placeholder="Brief description of the location"
                  value={reviewData.description}
                  onChange={(e) => setReviewData(prev => ({ ...prev, description: e.target.value }))}
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-200 mb-1">Review Text</label>
                <textarea
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  rows={4}
                  placeholder="Enter the review to analyse"
                  value={reviewData.review}
                  onChange={(e) => setReviewData(prev => ({ ...prev, review: e.target.value }))}
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-200 mb-1">
                  Rating: {reviewData.rating}/5
                </label>
                <input
                  type="range"
                  min="1"
                  max="5"
                  className="w-full h-2 bg-gray-600 rounded-lg appearance-none cursor-pointer slider"
                  value={reviewData.rating}
                  onChange={(e) => setReviewData(prev => ({ ...prev, rating: parseInt(e.target.value) }))}
                />
                <div className="flex justify-between text-xs text-gray-400 mt-1">
                  <span>1</span>
                  <span>2</span>
                  <span>3</span>
                  <span>4</span>
                  <span>5</span>
                </div>
              </div>
            </div>
          </div>

          <div className="mt-6 flex justify-center">
            <button
              onClick={handleAnalyze}
              disabled={isAnalyzing}
              className="px-6 py-3 bg-gray-700 text-white rounded-md hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed flex items-center transition-all duration-300 group overflow-hidden relative"
            >
              {isAnalyzing ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                  Analysing...
                </>
              ) : (
                <>
                  <span className="transition-transform duration-300 group-hover:-translate-x-3">Send Review</span>
                  <Send className="w-4 h-4 absolute right-4 translate-x-8 opacity-0 transition-all duration-300 group-hover:translate-x-0 group-hover:opacity-100" />
                </>
              )}
            </button>
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="bg-red-900/50 border border-red-700 rounded-md p-4 mb-6">
            <div className="flex items-center">
              <AlertCircle className="w-5 h-5 text-red-400 mr-2" />
              <p className="text-red-200">{error}</p>
            </div>
          </div>
        )}

        {/* Results */}
        {logs.length > 0 && (
          <div className="bg-gray-800 rounded-xl shadow-lg p-6">
            <h2 className="text-xl font-semibold text-white mb-4">Analysis Results</h2>
            <div className="space-y-3">
              {logs.map((log, index) => (
                <div
                  key={index}
                  className={`flex items-start gap-3 p-3 rounded-md border ${getLogStyle(log.type)}`}
                >
                  {getLogIcon(log.type)}
                  <p className="text-sm flex-1">{log.message}</p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
