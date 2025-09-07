'use client';

import { useState } from 'react';
import { MapPin } from 'lucide-react';
import { GoogleMap, LoadScript, Marker, Autocomplete } from '@react-google-maps/api';

const libraries: ("places")[] = ["places"];

interface MapLocation {
  lat: number;
  lng: number;
  name?: string;
}

interface LocationSelectorProps {
  location: MapLocation | null;
  onLocationChange: (location: MapLocation | null, category: string, description: string) => void;
  searchValue: string;
  onSearchValueChange: (value: string) => void;
}

const mapContainerStyle = {
  width: '100%',
  height: '300px',
  borderRadius: '8px'
};

// Dark mode map styles
const darkMapStyles = [
  { elementType: "geometry", stylers: [{ color: "#242f3e" }] },
  { elementType: "labels.text.stroke", stylers: [{ color: "#242f3e" }] },
  { elementType: "labels.text.fill", stylers: [{ color: "#746855" }] },
  {
    featureType: "administrative.locality",
    elementType: "labels.text.fill",
    stylers: [{ color: "#d59563" }],
  },
  {
    featureType: "poi",
    elementType: "labels.text.fill",
    stylers: [{ color: "#d59563" }],
  },
  {
    featureType: "poi.park",
    elementType: "geometry",
    stylers: [{ color: "#263c3f" }],
  },
  {
    featureType: "poi.park",
    elementType: "labels.text.fill",
    stylers: [{ color: "#6b9a76" }],
  },
  {
    featureType: "road",
    elementType: "geometry",
    stylers: [{ color: "#38414e" }],
  },
  {
    featureType: "road",
    elementType: "geometry.stroke",
    stylers: [{ color: "#212a37" }],
  },
  {
    featureType: "road",
    elementType: "labels.text.fill",
    stylers: [{ color: "#9ca5b3" }],
  },
  {
    featureType: "road.highway",
    elementType: "geometry",
    stylers: [{ color: "#746855" }],
  },
  {
    featureType: "road.highway",
    elementType: "geometry.stroke",
    stylers: [{ color: "#1f2835" }],
  },
  {
    featureType: "road.highway",
    elementType: "labels.text.fill",
    stylers: [{ color: "#f3d19c" }],
  },
  {
    featureType: "transit",
    elementType: "geometry",
    stylers: [{ color: "#2f3948" }],
  },
  {
    featureType: "transit.station",
    elementType: "labels.text.fill",
    stylers: [{ color: "#d59563" }],
  },
  {
    featureType: "water",
    elementType: "geometry",
    stylers: [{ color: "#17263c" }],
  },
  {
    featureType: "water",
    elementType: "labels.text.fill",
    stylers: [{ color: "#515c6d" }],
  },
  {
    featureType: "water",
    elementType: "labels.text.stroke",
    stylers: [{ color: "#17263c" }],
  },
];

const defaultCenter = {
  lat: 1.3521, // Singapore default
  lng: 103.8198
};

// Utility functions
const formatPlaceType = (type: string) => type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());

const createDescription = (name?: string, vicinity?: string) => {
  if (name && vicinity) return `${name} located in ${vicinity}`;
  if (name) return name;
  if (vicinity) return `Location in ${vicinity}`;
  return '';
};

export default function LocationSelector({ location, onLocationChange, searchValue, onSearchValueChange }: LocationSelectorProps) {
  const [autocomplete, setAutocomplete] = useState<google.maps.places.Autocomplete | null>(null);
  const [map, setMap] = useState<google.maps.Map | null>(null);
  const [lastClickTime, setLastClickTime] = useState<number>(0);

  const onLoad = (autocomplete: google.maps.places.Autocomplete) => {
    setAutocomplete(autocomplete);
  };

  const onMapLoad = (mapInstance: google.maps.Map) => {
    setMap(mapInstance);
    
    // Disable default info windows by adding click listener
    mapInstance.addListener('click', (event: google.maps.MapMouseEvent) => {
      // Close any open info windows
      const infoWindow = new google.maps.InfoWindow();
      infoWindow.close();
    });
  };

  const onPlaceChanged = () => {
    if (autocomplete !== null) {
      const place = autocomplete.getPlace();
      if (place.geometry?.location) {
        const lat = place.geometry.location.lat();
        const lng = place.geometry.location.lng();
        const placeName = place.formatted_address || place.name || `${lat.toFixed(4)}, ${lng.toFixed(4)}`;
        
        // Auto-fill category from place types
        const category = place.types?.[0] ? formatPlaceType(place.types[0]) : '';
        
        // Auto-fill description from available place data
        const description = createDescription(place.name, place.vicinity);
        
        onLocationChange({ lat, lng, name: placeName }, category, description);
        onSearchValueChange(placeName);
      }
    }
  };

  const handleMapClick = (event: google.maps.MapMouseEvent) => {
    if (event.latLng && map) {
      // Debounce clicks to reduce API calls (prevent rapid clicking)
      const now = Date.now();
      if (now - lastClickTime < 1000) { // 1 second cooldown
        return;
      }
      setLastClickTime(now);

      const lat = event.latLng.lat();
      const lng = event.latLng.lng();
      const clickedLocation = { lat, lng };
      
      // Search for nearby places within 25m radius
      const service = new google.maps.places.PlacesService(map);
      const request: google.maps.places.PlaceSearchRequest = {
        location: clickedLocation,
        radius: 25, // 25 meters - much more precise
        type: 'establishment' // Only businesses/establishments
      };
      
      service.nearbySearch(request, (results, status) => {
        if (status === google.maps.places.PlacesServiceStatus.OK && results && results.length > 0) {
          // Filter out non-business types and find actual establishments
          const businessTypes = ['restaurant', 'store', 'lodging', 'gas_station', 'bank', 'hospital', 'pharmacy', 'shopping_mall', 'gym', 'beauty_salon', 'car_repair', 'food', 'meal_takeaway', 'meal_delivery'];
          const excludedTypes = ['locality', 'political', 'country', 'administrative_area_level_1', 'administrative_area_level_2', 'postal_code', 'route', 'street_address', 'neighborhood', 'sublocality'];
          
          const actualBusiness = results.find(place => 
            place.types?.some(type => businessTypes.includes(type)) &&
            !place.types?.every(type => excludedTypes.includes(type))
          );
          
          if (actualBusiness) {
            // Auto-fill with business data
            const businessType = actualBusiness.types?.find(type => businessTypes.includes(type)) || actualBusiness.types?.[0];
            const category = businessType ? formatPlaceType(businessType) : '';
            const description = createDescription(actualBusiness.name, actualBusiness.vicinity);
            
            const locationName = actualBusiness.name || actualBusiness.vicinity || `${lat.toFixed(4)}, ${lng.toFixed(4)}`;
            onLocationChange({ lat, lng, name: locationName }, category, description);
            onSearchValueChange(locationName);
            return;
          }
        }
        
        // No businesses found - just use coordinates without geocoding API call
        const placeName = `${lat.toFixed(4)}, ${lng.toFixed(4)}`;
        onLocationChange({ lat, lng, name: placeName }, '', '');
        onSearchValueChange(placeName);
      });
    }
  };

  return (
    <div>
      <label className="block text-sm font-medium text-gray-200 mb-2">
        <MapPin className="inline w-4 h-4 mr-1" />
        Select Location
      </label>
      <LoadScript 
        googleMapsApiKey={process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY || ''} 
        libraries={libraries}
      >
        <div className="space-y-3">
          {/* Search Box */}
          <Autocomplete onLoad={onLoad} onPlaceChanged={onPlaceChanged}>
            <input
              type="text"
              placeholder="Search for a location..."
              value={searchValue}
              onChange={(e) => onSearchValueChange(e.target.value)}
              className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </Autocomplete>
          
          <div className="text-center">
            <span className="text-sm text-gray-400">Search for best auto-fill, or click map to select coordinates</span>
          </div>
          
          {/* Map */}
          <GoogleMap
            mapContainerStyle={mapContainerStyle}
            center={location || defaultCenter}
            zoom={location ? 15 : 11}
            onClick={handleMapClick}
            onLoad={onMapLoad}
            options={{
              styles: darkMapStyles,
              disableDefaultUI: true,
              zoomControl: true,
              streetViewControl: false,
              mapTypeControl: false,
              fullscreenControl: false,
              mapTypeId: 'roadmap',
              maxZoom: 21,
              minZoom: 11,
              scrollwheel: true,
              restriction: {
                latLngBounds: {
                  north: 1.48,
                  south: 1.20,
                  west: 103.58,
                  east: 104.08
                },
                strictBounds: true
              }
            }}
          >
            {location && <Marker 
              position={location} 
              options={{ 
                clickable: false,
                draggable: false 
              }}
            />}
          </GoogleMap>
        </div>
      </LoadScript>
      {location && (
        <p className="text-sm text-gray-300 mt-2">
          Selected: {location.name || `${location.lat.toFixed(4)}, ${location.lng.toFixed(4)}`}
        </p>
      )}
    </div>
  );
}