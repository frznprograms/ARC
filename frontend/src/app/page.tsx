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


interface StageUpdate {
  stage: number;
  status: 'starting' | 'passed' | 'rejected' | 'error' | 'banned' | 'uncertain';
  message: string;
  scores?: {
    ad: number;
    irrelevant: number;
    rant: number;
    unsafe: number;
  };
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
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchValue, setSearchValue] = useState('');
  const [currentStage, setCurrentStage] = useState<number>(0);
  const [stageUpdates, setStageUpdates] = useState<StageUpdate[]>([]);
  const [stageCounters, setStageCounters] = useState({
    safety_stage: 0,
    fasttext_stage: 0,
    encoder_stage: 0
  });

  const handleLocationChange = (newLocation: MapLocation | null, category: string, description: string) => {
    setLocation(newLocation);
    setReviewData(prev => ({ 
      ...prev, 
      name: newLocation?.name || '',
      category,
      description
    }));
  };

  const fetchStageCounters = async () => {
    try {
      const response = await fetch('http://127.0.0.1:8000/stage_counters/');
      if (response.ok) {
        const counters = await response.json();
        setStageCounters(counters);
      }
    } catch (error) {
      console.error('Failed to fetch stage counters:', error);
    }
  };

  const handleAnalyze = async () => {
    if (!reviewData.category || !reviewData.description || !reviewData.review || !location) {
      setError('Please fill in all fields and select a location');
      return;
    }

    setIsAnalyzing(true);
    setError(null);
    setCurrentStage(0);
    setStageUpdates([]);
    
    // Fetch initial counters
    await fetchStageCounters();

    try {
      const response = await fetch('http://127.0.0.1:8000/analyze_review_stream/', {
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

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('Failed to get response reader');
      }

      const decoder = new TextDecoder();
      
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              const update: StageUpdate = {
                stage: data.stage,
                status: data.status,
                message: data.message,
                scores: data.scores
              };
              
              setStageUpdates(prev => [...prev, update]);
              setCurrentStage(data.stage);
              
              // Enhanced logging for threshold tuning
              if (data.scores) {
                console.log('=== ENCODER PROBABILITIES FOR THRESHOLD TUNING ===');
                console.log('Review:', reviewData.review.slice(0, 100) + '...');
                console.log('Probabilities:', data.scores);
                
                const thresholds = { ad: 0.3, irrelevant: 0.25, rant: 0.2, unsafe: 0.4 };
                const triggered = Object.entries(data.scores).filter(([key, prob]) => 
                  prob > thresholds[key as keyof typeof thresholds]
                );
                
                console.log('Triggered categories:', triggered.length > 0 ? triggered : 'None');
                console.log('Result:', data.status);
                console.log('===========================================');
              }
            } catch (parseError) {
              console.error('Error parsing SSE data:', parseError);
            }
          }
        }
      }
    } catch (err) {
      setError(`Connection error: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setIsAnalyzing(false);
      // Fetch updated counters after analysis
      await fetchStageCounters();
    }
  };


  const getStageIcon = (status: string) => {
    switch (status) {
      case 'passed':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'rejected':
        return <AlertCircle className="w-5 h-5 text-red-500" />;
      case 'error':
        return <AlertCircle className="w-5 h-5 text-red-500" />;
      case 'uncertain':
        return <AlertCircle className="w-5 h-5 text-orange-500" />;
      case 'banned':
        return <AlertCircle className="w-5 h-5 text-red-500" />;
      default:
        return <Clock className="w-5 h-5 text-gray-400" />;
    }
  };

  const getStageStyle = (status: string) => {
    switch (status) {
      case 'passed':
        return 'border-green-500 bg-green-900/20 text-green-200';
      case 'rejected':
        return 'border-red-500 bg-red-900/20 text-red-200';
      case 'error':
        return 'border-red-500 bg-red-900/20 text-red-200';
      case 'banned':
        return 'border-red-500 bg-red-900/20 text-red-200';
      case 'starting':
        return 'border-blue-500 bg-blue-900/20 text-blue-200';
      case 'uncertain':
        return 'border-orange-500 bg-orange-900/20 text-orange-200';
      default:
        return 'border-gray-600 bg-gray-800 text-gray-400';
    }
  };

  const getStageTitle = (stage: number) => {
    switch (stage) {
      case 1: return 'Safety Check';
      case 2: return 'Fasttext Classification';
      case 3: return 'Encoder Analysis';
      default: return 'Unknown Stage';
    }
  };

  const getStageCounter = (stage: number) => {
    switch (stage) {
      case 1: return stageCounters.safety_stage;
      case 2: return stageCounters.fasttext_stage;
      case 3: return stageCounters.encoder_stage;
      default: return 0;
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 p-6">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-white mb-2">ARC</h1>
          <p className="text-gray-300">Analyse location reviews using ML-powered detection</p>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
          {/* Left Column - Form */}
          <div className="xl:col-span-2">
            <div className="bg-gray-800 rounded-xl shadow-lg p-6 mb-6">
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
                      rows={2}
                      placeholder="Brief description of the location"
                      value={reviewData.description}
                      onChange={(e) => setReviewData(prev => ({ ...prev, description: e.target.value }))}
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

              <div className="mt-6">
                <label className="block text-sm font-medium text-gray-200 mb-1">Review Text</label>
                <textarea
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  rows={4}
                  placeholder="Enter the review to analyse"
                  value={reviewData.review}
                  onChange={(e) => setReviewData(prev => ({ ...prev, review: e.target.value }))}
                />
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
          </div>

          {/* Right Column - Analysis Progress */}
          <div className="xl:col-span-1">
            {/* Error Display */}
            {error && (
              <div className="bg-red-900/50 border border-red-700 rounded-md p-4 mb-6">
                <div className="flex items-center">
                  <AlertCircle className="w-5 h-5 text-red-400 mr-2" />
                  <p className="text-red-200 text-sm">{error}</p>
                </div>
              </div>
            )}

            {/* Stage Progress */}
            <div className="bg-gray-800 rounded-xl shadow-lg p-6 mb-6">
              <h2 className="text-lg font-semibold text-white mb-4">Analysis Progress</h2>
              <div className="space-y-4">
                {[1, 2, 3].map((stage) => {
                  // Get the LATEST update for this stage (not the first one)
                  const stageUpdates_forStage = stageUpdates.filter(u => u.stage === stage);
                  const stageUpdate = stageUpdates_forStage[stageUpdates_forStage.length - 1];
                  const isCompleted = stageUpdate && ['passed', 'rejected', 'error', 'banned', 'uncertain'].includes(stageUpdate.status);
                  const isCurrentlyProcessing = currentStage === stage && isAnalyzing && !isCompleted;
                  
                  // Determine the icon to show
                  let icon;
                  if (isCompleted) {
                    icon = getStageIcon(stageUpdate.status);
                  } else if (isCurrentlyProcessing) {
                    icon = <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-500"></div>;
                  } else {
                    icon = <Clock className="w-5 h-5 text-gray-400" />;
                  }
                  
                  return (
                    <div
                      key={stage}
                      className={`flex items-center gap-3 p-3 rounded-md border transition-all duration-300 ${
                        stageUpdate ? getStageStyle(stageUpdate.status) : 'border-gray-600 bg-gray-800 text-gray-400'
                      }`}
                    >
                      {icon}
                      <div className="flex-1">
                        <div className="flex items-center justify-between">
                          <div className="font-medium text-sm">{getStageTitle(stage)}</div>
                          <div className="text-xs opacity-60">
                            {getStageCounter(stage)} runs
                          </div>
                        </div>
                        {stageUpdate && stageUpdate.message && (
                          <div className="text-xs mt-1 opacity-80">{stageUpdate.message}</div>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Final Result */}
            {!isAnalyzing && stageUpdates.length > 0 && (
              <div className="bg-gray-800 rounded-xl shadow-lg p-6">
                <h2 className="text-lg font-semibold text-white mb-4">Final Result</h2>
                {(() => {
                  const finalUpdate = stageUpdates[stageUpdates.length - 1];
                  const passedCount = stageUpdates.slice(1).filter(u => u.status === 'passed').length;
                  const allPassed = passedCount >= 2;
                  const bannedUpdate = stageUpdates.find(u => u.status === 'banned');
                  if (bannedUpdate) {
                    return (
                      <div className="flex items-center gap-3 p-4 rounded-md border border-red-500 bg-red-900/20">
                        <AlertCircle className="w-6 h-6 text-red-500" />
                        <div>
                          <div className="font-medium text-red-200">User Banned</div>
                          <div className="text-sm text-red-300 mt-1">Account flagged for policy violations</div>
                          <div className="text-xs text-red-400 mt-1">{bannedUpdate.message}</div>
                        </div>
                      </div>
                    );
                  } else if (allPassed) {
                    return (
                      <div className="flex items-center gap-3 p-4 rounded-md border border-green-500 bg-green-900/20">
                        <CheckCircle className="w-6 h-6 text-green-500" />
                        <div>
                          <div className="font-medium text-green-200">Review Accepted</div>
                          <div className="text-sm text-green-300 mt-1">Passed all validation stages</div>
                        </div>
                      </div>
                    );
                  } else {
                    const rejectedStage = stageUpdates.find(u => u.status === 'rejected' || u.status === 'error');
                    return (
                      <div className="flex items-center gap-3 p-4 rounded-md border border-red-500 bg-red-900/20">
                        <AlertCircle className="w-6 h-6 text-red-500" />
                        <div>
                          <div className="font-medium text-red-200">Review Rejected</div>
                          <div className="text-sm text-red-300 mt-1">
                            {rejectedStage ? `Failed at ${getStageTitle(rejectedStage.stage)}` : 'Processing failed'}
                          </div>
                          {rejectedStage && (
                            <div className="text-xs text-red-400 mt-1">{rejectedStage.message}</div>
                          )}
                        </div>
                      </div>
                    );
                  }
                })()}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
