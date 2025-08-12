#!/usr/bin/env python3
# Magic Animal: Alaskan King Crab
"""
WeeWX Surf & Fishing Forecast Extension Installer
Phase II: Local Surf & Fishing Forecast System

Copyright 2025 Shane Burkhardt
"""

import os
import json
import urllib.request
import urllib.parse
import urllib.error
import sys
import subprocess
import time
import yaml
import math
import curses
import textwrap
from typing import Dict, List, Optional, Any, Tuple

# CRITICAL: Correct import path for WeeWX 5.1
try:
    from weecfg.extension import ExtensionInstaller
    import weewx.manager
    import weewx
    import weeutil.logger
    log = weeutil.logger.logging.getLogger(__name__)
except ImportError:
    print("Error: This installer requires WeeWX 5.1 or later")
    sys.exit(1)

# CORE ICONS: Consistent with Phase I patterns
CORE_ICONS = {
    'navigation': '📍',    # Location/station selection
    'status': '✅',        # Success indicators  
    'warning': '⚠️',       # Warnings/issues
    'selection': '🔧'      # Configuration/selection
}

# REQUIRED: Loader function for WeeWX extension system
def loader():
    return SurfFishingInstaller()


class InstallationProgressManager:
    """Progress indicator for long operations"""
    
    def __init__(self):
        self.spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self.current_step = 0
        
    def show_step_progress(self, step_name, current=None, total=None):
        """Show progress for a step with optional counter"""
        if current is not None and total is not None:
            char = self.spinner_chars[current % len(self.spinner_chars)]
            print(f"\r  {char} {step_name}... {current}/{total}", end='', flush=True)
        else:
            print(f"  {step_name}...", end='', flush=True)
    
    def complete_step(self, step_name):
        """Mark step as complete"""
        print(f"\r  {step_name}... {CORE_ICONS['status']}")
        
    def show_error(self, step_name, error_msg):
        """Show step error"""
        print(f"\r  {step_name}... {CORE_ICONS['warning']}  {error_msg}")


class GRIBLibraryManager:
    """Detect GRIB processing libraries and enforce prerequisites"""
    
    def __init__(self):
        self.available_library = None
    
    def detect_grib_libraries(self):
        """Detect available GRIB processing libraries"""
        
        # Method 1: Try eccodes-python (preferred)
        try:
            import eccodes
            self.available_library = 'eccodes'
            return True
        except ImportError:
            pass
        
        # Method 2: Try pygrib (fallback)
        try:
            import pygrib
            self.available_library = 'pygrib'
            return True
        except ImportError:
            pass
        
        return False
    
    def install_grib_library(self):
        """Check GRIB library availability and exit if not found"""
        
        print(f"\n{CORE_ICONS['selection']} GRIB Processing Library Check")
        print("WaveWatch III forecast data requires GRIB file processing capability.")
        print()
        
        if self.detect_grib_libraries():
            print(f"  {CORE_ICONS['status']} GRIB library detected: {self.available_library}")
            return True
        else:
            print(f"  {CORE_ICONS['warning']} No GRIB library found")
            print()
            print("PREREQUISITE MISSING:")
            print("This extension requires pygrib or eccodes-python to process")
            print("WaveWatch III GRIB forecast data for beach-specific surf forecasts.")
            print()
            print("Installation options:")
            print("  1. Recommended (lightweight - 161MB):")
            print("     sudo apt-get install python3-grib")
            print("  2. Alternative (heavier - 500MB+):")
            print("     sudo apt-get install libeccodes0 python3-eccodes")
            print("  3. pip fallback:")
            print("     pip install pygrib")
            print()
            print("The python3-grib package provides pygrib with reasonable dependencies.")
            print("This enables 7km resolution WaveWatch III forecasts for your surf spots.")
            print()
            print("Please install a GRIB library and run the installer again.")
            print("See README.md for detailed installation instructions.")
            sys.exit(1)


class AtmosphericDataAnalyzer:
    """Analyze atmospheric data coverage for user guidance"""
    
    def __init__(self, phase_i_data):
        self.phase_i_data = phase_i_data
    
    def analyze_atmospheric_coverage(self, surf_spots, ndbc_stations):
        """Analyze atmospheric data coverage for user's spots"""
        
        analysis = {
            'total_spots': len(surf_spots),
            'buoy_covered_spots': 0,
            'station_recommended_spots': 0,
            'coverage_details': [],
            'station_recommended': False,
            'hybrid_recommended': False
        }
        
        for spot in surf_spots:
            spot_analysis = {
                'name': spot['name'],
                'lat': spot['latitude'],
                'lon': spot['longitude']
            }
            
            # Find nearest atmospheric NDBC buoy
            nearest_atmospheric = self._find_nearest_atmospheric_buoy(spot, ndbc_stations)
            
            if nearest_atmospheric and nearest_atmospheric['distance'] <= 25:
                spot_analysis['wind_coverage'] = 'buoy'
                spot_analysis['buoy_distance'] = nearest_atmospheric['distance']
                spot_analysis['buoy_id'] = nearest_atmospheric['id']
                analysis['buoy_covered_spots'] += 1
            else:
                spot_analysis['wind_coverage'] = 'station_recommended'
                spot_analysis['reason'] = 'No atmospheric buoys within 25 miles'
                analysis['station_recommended_spots'] += 1
            
            analysis['coverage_details'].append(spot_analysis)
        
        # Determine overall recommendation
        if analysis['station_recommended_spots'] > analysis['buoy_covered_spots']:
            analysis['station_recommended'] = True
        else:
            analysis['hybrid_recommended'] = True
        
        return analysis
    
    def _find_nearest_atmospheric_buoy(self, spot, ndbc_stations):
        """Find nearest NDBC buoy with atmospheric sensors"""
        
        atmospheric_buoys = []
        
        for station in ndbc_stations:
            # Check if buoy has atmospheric capabilities
            if self._has_atmospheric_sensors(station):
                distance = self._calculate_distance(
                    spot['latitude'], spot['longitude'],
                    station['lat'], station['lon']
                )
                atmospheric_buoys.append({
                    'id': station['id'],
                    'distance': distance,
                    'lat': station['lat'],
                    'lon': station['lon']
                })
        
        if atmospheric_buoys:
            return min(atmospheric_buoys, key=lambda x: x['distance'])
        return None
    
    def _has_atmospheric_sensors(self, station):
        """Check if NDBC station has atmospheric sensors"""
        # This would check station metadata for wind/pressure capabilities
        # For now, approximate based on station type/name patterns
        station_name = station.get('name', '').lower()
        return any(keyword in station_name for keyword in [
            'weather', 'meteorological', 'met', 'atmospheric'
        ]) or station.get('type') == 'weather_buoy'
    
    def _calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two points in miles"""
        R = 3959  # Earth radius in miles
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_lat/2) * math.sin(delta_lat/2) +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(delta_lon/2) * math.sin(delta_lon/2))
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c


class SurfFishingConfigurator:
    """Interactive configuration for surf and fishing forecasts"""
    
    def __init__(self, config_dict, yaml_data):
        self.config_dict = config_dict
        self.yaml_data = yaml_data
        self.progress = InstallationProgressManager()
        self.grib_manager = GRIBLibraryManager()
        
    def run_interactive_setup(self):
        """Main configuration workflow"""
        
        print(f"{CORE_ICONS['navigation']} Surf & Fishing Forecast Configuration")
        print("="*60)
        print("Configure your personal surf and fishing forecast system")
        print("This extension reads data from Phase I and adds forecasting capabilities")
        print()
        
        # Step 1: Check Phase I dependency (ENHANCED)
        self._check_phase_i_dependency()
        
        # Step 2: Setup GRIB processing libraries (ENHANCED) 
        grib_available = self._setup_grib_processing()
        
        # Step 3: Configure data source strategy (ENHANCED)
        data_sources = self._configure_data_sources()
        
        # Step 4: Configure forecast types and locations (EXISTING PATTERN)
        forecast_types = self._select_location_types()
        selected_locations = {}
        
        if 'surf' in forecast_types:
            selected_locations['surf_spots'] = self._configure_surf_spots()
        
        if 'fishing' in forecast_types:
            selected_locations['fishing_spots'] = self._configure_fishing_spots()
        
        # Step 5: Analyze marine station integration (NEW)
        station_analysis = self._analyze_marine_station_integration(selected_locations)
        
        # Step 6: Create configuration dictionary (ENHANCED)
        config_dict = self._create_config_dict(forecast_types, data_sources, selected_locations, grib_available, station_analysis)
        
        # Step 7: Display final summary (NEW)
        self._display_configuration_summary(config_dict, station_analysis)
        
        return config_dict, selected_locations
    
    def _check_phase_i_dependency(self):
        """Verify Phase I marine data extension is installed"""
        
        print(f"{CORE_ICONS['selection']} Checking Phase I Dependency")
        
        # Check if Phase I service is configured
        if 'MarineDataService' not in self.config_dict:
            print(f"  {CORE_ICONS['warning']} Phase I Marine Data Extension not found")
            print("  This extension requires Phase I to be installed first")
            print("  Please install the Phase I marine data extension before continuing")
            sys.exit(1)
        
        print(f"  {CORE_ICONS['status']} Phase I Marine Data Extension detected")
    
    def _setup_grib_processing(self):
        """Setup GRIB processing libraries"""
        
        if self.grib_manager.detect_grib_libraries():
            print(f"  {CORE_ICONS['status']} GRIB library detected: {self.grib_manager.available_library}")
            return True
        else:
            return self.grib_manager.install_grib_library()
    
    def _configure_data_sources(self):
        """
        Configure atmospheric data strategy based on user's station location
        Uses simple question about coastal proximity for decision making
        """
        
        print(f"\n{CORE_ICONS['selection']} Station Location Assessment")
        print("To optimize your surf and fishing forecasts, we need to understand your station location.")
        print()
        
        # Get station coordinates for display (if available)
        station_lat = self.config_dict.get('Station', {}).get('latitude')
        station_lon = self.config_dict.get('Station', {}).get('longitude')
        
        if station_lat and station_lon:
            print(f"  Your WeeWX Station: {float(station_lat):.4f}, {float(station_lon):.4f}")
        
        print(f"  Location Question:")
        print()
        print("Is your weather station located within 5 miles of the ocean/coast?")
        print("(This determines whether your station data is useful for marine forecasting)")
        print()
        print("1. Yes - My station is within 5 miles of the coast")
        print("   → We'll use your station + NOAA data for best accuracy")
        print()
        print("2. No - My station is more than 5 miles inland") 
        print("   → We'll use NOAA marine data only for best accuracy")
        print()
        
        while True:
            try:
                choice = input("Is your station within 5 miles of the coast? (1=Yes, 2=No): ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nInstallation cancelled by user.")
                sys.exit(1)
            
            if choice == '1':
                print(f"\n  {CORE_ICONS['status']} DECISION: Hybrid Approach (Station + NOAA)")
                print(f"  Your coastal station will provide valuable local atmospheric data")
                print(f"      - Station: Local wind/pressure accuracy")
                print(f"      - NOAA: Marine wave/tide data")
                print(f"      - Combined: Best forecast accuracy")
                
                # Call existing station integration method
                return self._configure_station_integration()
                
            elif choice == '2':
                print(f"\n  {CORE_ICONS['status']} DECISION: NOAA-Only Approach")
                print(f"  NOAA marine buoys will provide the most accurate data")
                print(f"      - NOAA buoys: Best marine atmospheric data")
                print(f"      - Station: Too far inland for marine forecasts")
                
                return {
                    'type': 'noaa_only',
                    'reason': 'station_inland_user_selected'
                }
            else:
                print(f"  {CORE_ICONS['warning']} Please enter 1 for Yes or 2 for No")
    
    def _configure_station_integration(self):
        """
        Configure which station sensors to use (supplement with NOAA)
        Enhanced with location-based recommendations for coastal stations
        """
        
        print(f"\n{CORE_ICONS['selection']} Station Sensor Integration")
        print("Select which station sensors to use (supplement with NOAA):")
        print("Recommendations for coastal stations:")
        print()
        
        # Coastal station recommendations (since user selected coastal location)
        recommendations = {
            'wind': 'Highly recommended - excellent coastal wind representation',
            'pressure': 'Highly recommended - excellent for marine pressure systems', 
            'temperature': 'Recommended - good coastal air temperature'
        }
        
        available_sensors = {
            'wind': 'Wind speed and direction',
            'pressure': 'Barometric pressure', 
            'temperature': 'Air temperature'
        }
        
        selected_sensors = {}
        for sensor_key, description in available_sensors.items():
            recommendation = recommendations.get(sensor_key, 'Recommended')
            use_sensor = input(f"Use station {description}? ({recommendation}) (y/n, default n): ").strip().lower()
            selected_sensors[sensor_key] = use_sensor == 'y'
            
        return {
            'type': 'station_supplement',
            'sensors': selected_sensors,
            'reason': 'station_coastal_user_selected'
        }
    
    def _configure_locations(self):
        """Configure surf and fishing spots"""
        
        print(f"\n{CORE_ICONS['navigation']} Location Configuration")
        print("="*60)
        print("Configure your surf and fishing forecast locations")
        print("Maximum 10 locations total (surf + fishing combined)")
        print()
        
        locations = {
            'surf_spots': [],
            'fishing_spots': []
        }
        
        # Choose location types to configure
        location_types = self._select_location_types()
        
        # Configure each type
        if 'surf' in location_types:
            locations['surf_spots'] = self._configure_surf_spots()
            
        if 'fishing' in location_types:
            locations['fishing_spots'] = self._configure_fishing_spots()
        
        return locations
    
    def _select_location_types(self):
        """Select which types of locations to configure"""
        
        print("Select forecast types to configure:")
        print("1. Surf forecasts only")
        print("2. Fishing forecasts only") 
        print("3. Both surf and fishing forecasts")
        print()
        
        while True:
            choice = input("Select option (1-3): ").strip()
            if choice == '1':
                return ['surf']
            elif choice == '2':
                return ['fishing']
            elif choice == '3':
                return ['surf', 'fishing']
            else:
                print(f"{CORE_ICONS['warning']} Please enter 1, 2, or 3")
    
    def _configure_surf_spots(self):
        """Configure surf-specific locations"""
        
        surf_spots = []
        print(f"\n{CORE_ICONS['selection']} Surf Spot Configuration")
        print("Enter your surf spots (max 5 spots)")
        print()
        
        while len(surf_spots) < 5:
            print(f"Surf Spot {len(surf_spots) + 1}:")
            
            name = input("  Spot name (e.g., 'Malibu', 'Ocean Beach') [Enter to finish]: ").strip()
            if not name:
                break
                
            # Get coordinates
            lat, lon = self._get_coordinates_with_validation(name)
            if not lat or not lon:
                continue
            
            # Get surf-specific characteristics
            spot_config = self._configure_surf_characteristics(name, lat, lon)
            surf_spots.append(spot_config)
            
            print(f"  {CORE_ICONS['status']} Added surf spot: {name}")
        
        return surf_spots
    
    def _configure_fishing_spots(self):
        """Configure fishing-specific locations"""
        
        fishing_spots = []
        print(f"\n{CORE_ICONS['selection']} Fishing Spot Configuration")
        print("Enter your fishing spots (max 5 spots)")
        print()
        
        while len(fishing_spots) < 5:
            print(f"Fishing Spot {len(fishing_spots) + 1}:")
            
            name = input("  Spot name (e.g., 'Santa Monica Pier', 'Newport Harbor') [Enter to finish]: ").strip()
            if not name:
                break
                
            # Get coordinates
            lat, lon = self._get_coordinates_with_validation(name)
            if not lat or not lon:
                continue
            
            # Get fishing-specific characteristics
            spot_config = self._configure_fishing_characteristics(name, lat, lon)
            fishing_spots.append(spot_config)
            
            print(f"  {CORE_ICONS['status']} Added fishing spot: {name}")
        
        return fishing_spots
    
    def _get_coordinates_with_validation(self, location_name):
        """Get and validate coordinates for a location"""
        
        print(f"  Coordinates for {location_name}:")
        
        try:
            lat_str = input("    Latitude (decimal degrees, e.g., 34.0522): ").strip()
            lon_str = input("    Longitude (decimal degrees, e.g., -118.2437): ").strip()
            
            lat = float(lat_str)
            lon = float(lon_str)
            
            # Basic validation for US coastal waters
            if not (-180 <= lon <= 180):
                print(f"    {CORE_ICONS['warning']} Invalid longitude. Must be between -180 and 180")
                return None, None
                
            if not (-90 <= lat <= 90):
                print(f"    {CORE_ICONS['warning']} Invalid latitude. Must be between -90 and 90")
                return None, None
            
            return lat, lon
            
        except ValueError:
            print(f"    {CORE_ICONS['warning']} Invalid coordinates. Please enter decimal numbers")
            return None, None
    
    def _configure_surf_characteristics(self, name, lat, lon):
        """Configure surf-specific spot characteristics"""
        
        config = {
            'name': name,
            'latitude': lat,
            'longitude': lon,
            'type': 'surf'
        }
        
        # Bottom type affects wave breaking patterns
        print("  Bottom type (affects how waves break):")
        print("    1. Sand (Beach break)")
        print("    2. Reef (Coral/rock reef)")
        print("    3. Point break (Rocky point)")
        print("    4. Jetty/Pier (Man-made structure)")
        print("    5. Mixed")
        
        bottom_choice = input("  Select bottom type (1-5, default 1): ").strip() or "1"
        config['bottom_type'] = {
            "1": "sand", "2": "reef", "3": "point", "4": "jetty", "5": "mixed"
        }.get(bottom_choice, "sand")
        
        # Exposure to swell
        print("  Exposure to ocean swell:")
        print("    1. Fully exposed (open ocean)")
        print("    2. Semi-protected (bay/cove)")
        print("    3. Protected (harbor/inlet)")
        
        exposure_choice = input("  Select exposure (1-3, default 1): ").strip() or "1"
        config['exposure'] = {
            "1": "exposed", "2": "semi_protected", "3": "protected"
        }.get(exposure_choice, "exposed")
        
        return config
    
    def _configure_fishing_characteristics(self, name, lat, lon):
        """Configure fishing-specific spot characteristics"""
        
        config = {
            'name': name,
            'latitude': lat,
            'longitude': lon,
            'type': 'fishing'
        }
        
        # Location type affects fishing methods
        print("  Fishing location type:")
        print("    1. Shore/Beach (fishing from land)")
        print("    2. Pier/Jetty (fishing from structure)")
        print("    3. Boat (fishing from vessel)")
        print("    4. Mixed (multiple methods)")
        
        location_choice = input("  Select location type (1-4, default 1): ").strip() or "1"
        config['location_type'] = {
            "1": "shore", "2": "pier", "3": "boat", "4": "mixed"
        }.get(location_choice, "shore")
        
        # Target species category
        fish_categories = self.yaml_data.get('fish_categories', {})
        
        print("  Primary target fish category:")
        category_keys = list(fish_categories.keys())
        for i, (category_key, category_data) in enumerate(fish_categories.items(), 1):
            print(f"    {i}. {category_data['display_name']}")
        
        category_choice = input(f"  Select category (1-{len(fish_categories)}, default 1): ").strip() or "1"
        try:
            selected_category = category_keys[int(category_choice)-1]
        except (ValueError, IndexError):
            selected_category = category_keys[0] if category_keys else 'mixed_bag'
        
        config['target_category'] = selected_category
        
        return config
    
    def _create_config_dict(self, forecast_types, data_sources, selected_locations, grib_available, station_analysis):
        """
        Create configuration dictionary for WeeWX integration
        """
        
        # Build the main service configuration section
        config_dict = {
            'SurfFishingService': {
                'enable': 'true',
                'forecast_interval': '21600',  # 6 hours in seconds
                'log_success': 'false',
                'log_errors': 'true',
                'timeout': '60',
                'retry_attempts': '3',
                
                # NEW: Forecast configuration based on user selections
                'forecast_settings': {
                    'enabled_types': ','.join(forecast_types),
                    'forecast_hours': '72',  # 3-day forecasts
                    'rating_system': 'five_star',
                    'update_interval_hours': '6'
                },
                
                # EXPANDED: Data source configuration 
                'data_integration': {
                    'method': data_sources.get('type', 'noaa_only'),
                    'local_station_distance_km': str(station_analysis.get('distance_km', 999)),
                    'enable_station_data': 'true' if data_sources.get('type') == 'station_supplement' else 'false'
                },
                
                # PRESERVED: Station integration from original _transform_to_weewx_conf
                'station_integration': {
                    'type': data_sources['type']
                },
                
                # EXPANDED: GRIB processing capability 
                'grib_processing': {
                    'available': 'true' if grib_available else 'false',
                    'library': self.grib_manager.available_library if grib_available else 'none',
                    'download_timeout': '300',
                    'cache_hours': '6'
                },

                # PRESERVED: WaveWatch III endpoints from YAML (original functionality)
                'wavewatch_endpoints': self.yaml_data.get('api_endpoints', {}).get('wavewatch_iii', {}),
                
                # PRESERVED: Fish categories from YAML (original functionality)  
                'fish_categories': self.yaml_data.get('fish_categories', {}),
                
                # PRESERVED: User locations structure (original functionality)
                'surf_spots': {},
                'fishing_spots': {}
            }
        }
        
        # PRESERVED: Add station sensor configuration if using station integration (original logic)
        if data_sources['type'] == 'station_supplement':
            config_dict['SurfFishingService']['station_integration']['sensors'] = data_sources.get('sensors', {})
        
        # EXPANDED: Add surf spot configurations with proper handling
        if 'surf_spots' in selected_locations and selected_locations['surf_spots']:
            for i, spot in enumerate(selected_locations['surf_spots']):
                spot_key = f'spot_{i}'
                config_dict['SurfFishingService']['surf_spots'][spot_key] = {
                    'name': spot['name'],
                    'latitude': str(spot['latitude']),
                    'longitude': str(spot['longitude']),
                    'bottom_type': spot.get('bottom_type', 'sand'),
                    'exposure': spot.get('exposure', 'exposed')
                }
        
        # EXPANDED: Add fishing spot configurations with proper handling  
        if 'fishing_spots' in selected_locations and selected_locations['fishing_spots']:
            for i, spot in enumerate(selected_locations['fishing_spots']):
                spot_key = f'spot_{i}'
                config_dict['SurfFishingService']['fishing_spots'][spot_key] = {
                    'name': spot['name'],
                    'latitude': str(spot['latitude']),
                    'longitude': str(spot['longitude']),
                    'location_type': spot.get('location_type', 'shore'),
                    'target_category': spot.get('target_category', 'mixed_bag')
                }
        
        # NEW: Add station analysis results to configuration for service reference
        if station_analysis:
            config_dict['SurfFishingService']['station_analysis'] = {
                'analysis_completed': str(station_analysis.get('station_analysis_completed', False)),
                'accepted_recommendations': str(len(station_analysis.get('accepted_recommendations', []))),
                'coverage_quality': station_analysis.get('coverage_summary', {}).get('quality_score', 'unknown')
            }
        
        # NEW: Add forecast type tracking for service initialization
        config_dict['SurfFishingService']['forecast_types'] = forecast_types
        
        # NEW: Add data source tracking for service reference
        config_dict['SurfFishingService']['data_sources'] = {
            'type': data_sources['type'],
            'configured_at': str(int(time.time()))
        }
        
        return config_dict

    def _analyze_marine_station_integration(self, selected_locations):
        """
        Analyze Phase I marine station coverage for user locations
        Provide recommendations for optimization if needed
        """
        print(f"\n{CORE_ICONS['navigation']} Marine Station Integration Analysis")
        print("Analyzing your Phase I station coverage for optimal forecasting...")
        
        # Initialize analyzers
        phase_i_analyzer = PhaseIAnalyzer(self.config_dict)
        quality_analyzer = StationQualityAnalyzer(self.yaml_data)
        recommendation_engine = MarineStationRecommendationEngine(phase_i_analyzer, quality_analyzer)
        
        # Get Phase I coverage summary
        phase_i_coverage = phase_i_analyzer.analyze_phase_i_coverage()
        
        print(f"\nPhase I Station Summary:")
        print(f"  Total stations: {phase_i_coverage['total_stations']}")
        print(f"  CO-OPS tide stations: {phase_i_coverage['coops_count']}")
        print(f"  NDBC buoy stations: {phase_i_coverage['ndbc_count']}")
        
        if phase_i_coverage['distance_summary']['closest_station']:
            closest = phase_i_coverage['distance_summary']['closest_station']
            print(f"  Closest station: {closest['name']} ({closest['distance']:.1f} miles)")
        
        # Analyze all user locations
        all_user_locations = []
        for spot in selected_locations.get('surf_spots', []):
            all_user_locations.append(spot)
        for spot in selected_locations.get('fishing_spots', []):
            all_user_locations.append(spot)
        
        if not all_user_locations:
            print(f"\n{CORE_ICONS['status']} No location analysis needed - no locations configured")
            return
        
        # Generate recommendations
        recommendations = recommendation_engine.analyze_multi_location_optimization(all_user_locations)
        
        # Display current coverage summary
        current = recommendations['current_coverage']
        print(f"\nCoverage Quality Summary:")
        print(f"  Wave data quality: {current['wave_quality']:.1f}/1.0")
        print(f"  Atmospheric data quality: {current['atmospheric_quality']:.1f}/1.0")
        print(f"  Tide data quality: {current['tide_quality']:.1f}/1.0")
        
        # Present recommendations interactively
        accepted_recommendations = recommendation_engine.display_recommendations_interactive(recommendations)
        
        if accepted_recommendations:
            print(f"\n{CORE_ICONS['selection']} Implementation Options")
            print("Accepted recommendations will be noted in your configuration.")
            print("You can implement these manually in Phase I after installation.")
            print()
            
            # Store recommendations in configuration for reference
            return {
                'station_analysis_completed': True,
                'accepted_recommendations': accepted_recommendations,
                'coverage_summary': current
            }
        else:
            print(f"\n{CORE_ICONS['status']} Current configuration will be used as-is")
            return {
                'station_analysis_completed': True,
                'accepted_recommendations': [],
                'coverage_summary': current
            }

    def _display_configuration_summary(self, config_dict, station_analysis):
            """
            Display comprehensive configuration summary including station analysis
            Provides clear overview of what was configured
            """
            print(f"\n{CORE_ICONS['status']} Configuration Summary")
            print("="*50)
            
            service_config = config_dict['SurfFishingService']
            
            # Forecast types configured
            forecast_types = service_config['forecast_types']
            print(f"Forecast Types: {', '.join(forecast_types).title()}")
            
            # Data sources
            data_source_type = service_config['data_sources']['type']
            print(f"Data Strategy: {data_source_type.replace('_', ' ').title()}")
            
            # Location counts
            surf_count = len(service_config['surf_spots'])
            fishing_count = len(service_config['fishing_spots'])
            print(f"Locations: {surf_count} surf spots, {fishing_count} fishing spots")
            
            # GRIB processing
            grib_enabled = service_config['grib_processing']['available']
            grib_lib = service_config['grib_processing']['library']
            print(f"WaveWatch III: {'Enabled' if grib_enabled == 'true' else 'Disabled'}")
            if grib_enabled == 'true':
                print(f"  GRIB Library: {grib_lib}")
            
            # Station integration analysis results (FIXED)
            if station_analysis and 'station_analysis' in service_config:
                analysis_config = service_config['station_analysis']
                print(f"\nStation Integration Analysis:")
                print(f"  Analysis completed: {analysis_config['analysis_completed']}")
                
                if analysis_config.get('coverage_quality'):
                    print(f"  Coverage quality: {analysis_config['coverage_quality']}")
                
                recommendations_count = analysis_config.get('accepted_recommendations', '0')
                print(f"  Accepted recommendations: {recommendations_count}")
            
            # Data integration settings
            if 'data_integration' in service_config:
                data_integration = service_config['data_integration']
                print(f"\nData Integration:")
                print(f"  Method: {data_integration['method']}")
                print(f"  Station data enabled: {data_integration['enable_station_data']}")
                
                distance_km = data_integration.get('local_station_distance_km', 'unknown')
                if distance_km != '999':
                    print(f"  Station distance: {distance_km} km")
            
            print(f"\n{CORE_ICONS['status']} Configuration complete - ready for installation!")

    def _handle_configuration_error(self, error, context="configuration"):
        """
        Handle configuration errors gracefully with helpful messaging
        Provides actionable error information to users
        """
        error_msg = str(error).lower()
        
        if "phase i" in error_msg or "marinedataservice" in error_msg:
            print(f"\n{CORE_ICONS['warning']} Phase I Dependency Error")
            print("The Phase I Marine Data Extension is required but not properly configured.")
            print("\nPlease ensure Phase I is installed and has marine stations selected:")
            print("1. Check if Phase I is installed: weectl extension list")
            print("2. Reconfigure Phase I if needed: sudo weectl extension reconfigure marine_data") 
            print("3. Restart WeeWX: sudo systemctl restart weewx")
            print("4. Then retry this installation")
            
        elif "grib" in error_msg or "eccodes" in error_msg or "pygrib" in error_msg:
            print(f"\n{CORE_ICONS['warning']} GRIB Processing Error")
            print("GRIB library installation or configuration failed.")
            print("\nYou can continue without WaveWatch III forecasts,")
            print("using only Phase I marine data for forecasting.")
            print("\nTo fix GRIB processing later:")
            print("  sudo apt-get install eccodes python3-eccodes")
            print("  # OR")
            print("  pip3 install eccodes-python")
            
        elif "permission" in error_msg or "sudo" in error_msg:
            print(f"\n{CORE_ICONS['warning']} Permission Error")
            print("Installation requires administrator privileges.")
            print("\nPlease run the installer with sudo:")
            print("  sudo weectl extension install weewx-surf-fishing-1.0.0-alpha.zip")
            
        elif "network" in error_msg or "connection" in error_msg:
            print(f"\n{CORE_ICONS['warning']} Network Error")
            print("Cannot connect to required services during installation.")
            print("\nPlease check your internet connection and try again.")
            print("Installation can continue with limited functionality.")
            
        else:
            print(f"\n{CORE_ICONS['warning']} Configuration Error")
            print(f"An error occurred during {context}: {error}")
            print("\nYou may need to:")
            print("1. Check system requirements")
            print("2. Verify WeeWX 5.1+ is installed")
            print("3. Ensure proper permissions")
            print("4. Check available disk space")
        
        print()

class PhaseIAnalyzer:
    """
    Analyze existing Phase I station configuration from CONF metadata
    Provides zero-API station analysis for Phase II optimization
    """
    
    def __init__(self, config_dict):
        """
        Initialize analyzer with WeeWX configuration dictionary
        """
        self.config_dict = config_dict
        self.marine_config = config_dict.get('MarineDataService', {})
        self.station_metadata = self.marine_config.get('station_metadata', {})
        
    def get_available_stations(self):
        """
        Extract all Phase I selected stations with metadata from CONF
        Returns comprehensive station data without any API calls
        """
        stations = {
            'coops_stations': [],
            'ndbc_stations': [],
            'total_count': 0
        }
        
        # Process CO-OPS stations from Phase I metadata
        coops_metadata = self.station_metadata.get('coops_stations', {})
        for station_id, metadata in coops_metadata.items():
            station_info = {
                'station_id': station_id,
                'name': metadata.get('name', f'Station {station_id}'),
                'latitude': float(metadata.get('latitude', 0)),
                'longitude': float(metadata.get('longitude', 0)),
                'distance_miles': float(metadata.get('distance_miles', 0)),
                'capabilities': metadata.get('capabilities', '').split(', '),
                'station_type': metadata.get('station_type', 'coops'),
                'has_tide_data': 'water_level' in metadata.get('capabilities', '') or 'predictions' in metadata.get('capabilities', ''),
                'has_water_temp': 'water_temperature' in metadata.get('capabilities', '')
            }
            stations['coops_stations'].append(station_info)
        
        # Process NDBC stations from Phase I metadata  
        ndbc_metadata = self.station_metadata.get('ndbc_stations', {})
        for station_id, metadata in ndbc_metadata.items():
            capabilities = {
                'wave_data': metadata.get('wave_capability', 'false') == 'true',
                'atmospheric_data': metadata.get('atmospheric_capability', 'false') == 'true'
            }
            
            station_info = {
                'station_id': station_id,
                'name': metadata.get('name', f'Buoy {station_id}'),
                'latitude': float(metadata.get('latitude', 0)),
                'longitude': float(metadata.get('longitude', 0)),
                'distance_miles': float(metadata.get('distance_miles', 0)),
                'capabilities': capabilities,
                'station_type': metadata.get('station_type', 'ndbc'),
                'has_wave_data': capabilities['wave_data'],
                'has_atmospheric_data': capabilities['atmospheric_data']
            }
            stations['ndbc_stations'].append(station_info)
        
        stations['total_count'] = len(stations['coops_stations']) + len(stations['ndbc_stations'])
        return stations

    def analyze_phase_i_coverage(self):
        """
        Analyze Phase I station coverage for basic metrics
        Returns coverage summary for user information
        """
        stations = self.get_available_stations()
        
        analysis = {
            'total_stations': stations['total_count'],
            'coops_count': len(stations['coops_stations']),
            'ndbc_count': len(stations['ndbc_stations']),
            'coverage_types': set(),
            'distance_summary': {
                'closest_station': None,
                'average_distance': 0,
                'farthest_station': None
            }
        }
        
        all_stations = stations['coops_stations'] + stations['ndbc_stations']
        
        if all_stations:
            # Calculate distance statistics
            distances = [s['distance_miles'] for s in all_stations]
            analysis['distance_summary']['average_distance'] = sum(distances) / len(distances)
            
            closest = min(all_stations, key=lambda s: s['distance_miles'])
            analysis['distance_summary']['closest_station'] = {
                'name': closest['name'],
                'distance': closest['distance_miles'],
                'type': closest['station_type']
            }
            
            farthest = max(all_stations, key=lambda s: s['distance_miles'])
            analysis['distance_summary']['farthest_station'] = {
                'name': farthest['name'], 
                'distance': farthest['distance_miles'],
                'type': farthest['station_type']
            }
            
            # Identify coverage types
            for station in stations['coops_stations']:
                if station['has_tide_data']:
                    analysis['coverage_types'].add('tide_data')
                if station['has_water_temp']:
                    analysis['coverage_types'].add('water_temperature')
            
            for station in stations['ndbc_stations']:
                if station['has_wave_data']:
                    analysis['coverage_types'].add('wave_data')
                if station['has_atmospheric_data']:
                    analysis['coverage_types'].add('atmospheric_data')
        
        return analysis
    

class StationQualityAnalyzer:
    """
    Analyze station data quality for user locations using research-based thresholds
    Implements coastal oceanography research for distance-based quality scoring
    """
    
    def __init__(self, yaml_data):
        """
        Initialize with quality thresholds from YAML configuration
        """
        # Load research-based quality thresholds from YAML
        quality_config = yaml_data.get('station_quality_thresholds', {})
        
        self.wave_thresholds = quality_config.get('wave_data', {
            'excellent_distance_km': 40,  # 25 miles
            'good_distance_km': 80,       # 50 miles  
            'fair_distance_km': 160,      # 100 miles
            'minimum_quality': 0.6
        })
        
        self.atmospheric_thresholds = quality_config.get('atmospheric_data', {
            'excellent_distance_km': 80,   # 50 miles
            'good_distance_km': 160,       # 100 miles
            'fair_distance_km': 320,       # 200 miles
            'minimum_quality': 0.5
        })
        
        self.tide_thresholds = quality_config.get('tide_data', {
            'excellent_distance_km': 80,   # 50 miles
            'good_distance_km': 160,       # 100 miles  
            'fair_distance_km': 240,       # 150 miles
            'minimum_quality': 0.7
        })
    
    def calculate_wave_quality_score(self, distance_miles):
        """
        Calculate wave data quality score based on research-validated distance thresholds
        Returns quality score (0.0-1.0) and quality level
        """
        distance_km = distance_miles * 1.60934
        
        if distance_km <= self.wave_thresholds['excellent_distance_km']:
            return 1.0, 'excellent'
        elif distance_km <= self.wave_thresholds['good_distance_km']:
            return 0.8, 'good'
        elif distance_km <= self.wave_thresholds['fair_distance_km']:
            return 0.6, 'fair'
        else:
            # Linear decay beyond fair threshold
            max_distance = self.wave_thresholds['fair_distance_km'] * 2
            if distance_km >= max_distance:
                return 0.1, 'poor'
            decay_factor = (max_distance - distance_km) / max_distance
            return max(0.1, decay_factor * 0.5), 'poor'
    
    def calculate_atmospheric_quality_score(self, distance_miles):
        """
        Calculate atmospheric data quality score based on research thresholds
        Returns quality score (0.0-1.0) and quality level
        """
        distance_km = distance_miles * 1.60934
        
        if distance_km <= self.atmospheric_thresholds['excellent_distance_km']:
            return 1.0, 'excellent'
        elif distance_km <= self.atmospheric_thresholds['good_distance_km']:
            return 0.8, 'good'
        elif distance_km <= self.atmospheric_thresholds['fair_distance_km']:
            return 0.6, 'fair'
        else:
            max_distance = self.atmospheric_thresholds['fair_distance_km'] * 2
            if distance_km >= max_distance:
                return 0.1, 'poor'
            decay_factor = (max_distance - distance_km) / max_distance
            return max(0.1, decay_factor * 0.5), 'poor'
    
    def calculate_tide_quality_score(self, distance_miles):
        """
        Calculate tide data quality score based on coastal research
        Returns quality score (0.0-1.0) and quality level
        """
        distance_km = distance_miles * 1.60934
        
        if distance_km <= self.tide_thresholds['excellent_distance_km']:
            return 1.0, 'excellent'
        elif distance_km <= self.tide_thresholds['good_distance_km']:
            return 0.8, 'good'
        elif distance_km <= self.tide_thresholds['fair_distance_km']:
            return 0.6, 'fair'
        else:
            max_distance = self.tide_thresholds['fair_distance_km'] * 2
            if distance_km >= max_distance:
                return 0.1, 'poor'
            decay_factor = (max_distance - distance_km) / max_distance
            return max(0.1, decay_factor * 0.5), 'poor'

    def analyze_location_coverage(self, location, available_stations):
        """
        Analyze data coverage quality for specific user location
        Returns comprehensive coverage analysis with quality scores
        """
        location_lat = location['latitude']
        location_lon = location['longitude']
        location_name = location['name']
        
        analysis = {
            'location': location_name,
            'coordinates': (location_lat, location_lon),
            'wave_sources': [],
            'atmospheric_sources': [],
            'tide_sources': [],
            'quality_summary': {
                'wave_quality': 0.0,
                'atmospheric_quality': 0.0,
                'tide_quality': 0.0,
                'overall_quality': 0.0
            },
            'recommendations': []
        }
        
        # Analyze NDBC stations for wave and atmospheric data
        for station in available_stations['ndbc_stations']:
            station_distance = self._calculate_distance(
                location_lat, location_lon,
                station['latitude'], station['longitude']
            )
            
            if station['has_wave_data']:
                wave_score, wave_level = self.calculate_wave_quality_score(station_distance)
                analysis['wave_sources'].append({
                    'station_id': station['station_id'],
                    'name': station['name'],
                    'distance_miles': station_distance,
                    'quality_score': wave_score,
                    'quality_level': wave_level
                })
            
            if station['has_atmospheric_data']:
                atmo_score, atmo_level = self.calculate_atmospheric_quality_score(station_distance)
                analysis['atmospheric_sources'].append({
                    'station_id': station['station_id'],
                    'name': station['name'],
                    'distance_miles': station_distance,
                    'quality_score': atmo_score,
                    'quality_level': atmo_level
                })
        
        # Analyze CO-OPS stations for tide data
        for station in available_stations['coops_stations']:
            if station['has_tide_data']:
                station_distance = self._calculate_distance(
                    location_lat, location_lon,
                    station['latitude'], station['longitude']
                )
                
                tide_score, tide_level = self.calculate_tide_quality_score(station_distance)
                analysis['tide_sources'].append({
                    'station_id': station['station_id'],
                    'name': station['name'],
                    'distance_miles': station_distance,
                    'quality_score': tide_score,
                    'quality_level': tide_level
                })
        
        # Calculate best quality scores for each data type
        if analysis['wave_sources']:
            analysis['quality_summary']['wave_quality'] = max(s['quality_score'] for s in analysis['wave_sources'])
        
        if analysis['atmospheric_sources']:
            analysis['quality_summary']['atmospheric_quality'] = max(s['quality_score'] for s in analysis['atmospheric_sources'])
        
        if analysis['tide_sources']:
            analysis['quality_summary']['tide_quality'] = max(s['quality_score'] for s in analysis['tide_sources'])
        
        # Calculate overall quality (weighted average)
        weights = {'wave': 0.4, 'atmospheric': 0.4, 'tide': 0.2}
        overall_quality = (
            analysis['quality_summary']['wave_quality'] * weights['wave'] +
            analysis['quality_summary']['atmospheric_quality'] * weights['atmospheric'] +
            analysis['quality_summary']['tide_quality'] * weights['tide']
        )
        analysis['quality_summary']['overall_quality'] = overall_quality
        
        return analysis
    
    def _calculate_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate distance between two points using haversine formula
        Returns distance in miles
        """
        import math
        R = 3959  # Earth radius in miles
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_lat/2) * math.sin(delta_lat/2) +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(delta_lon/2) * math.sin(delta_lon/2))
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    

class MarineStationRecommendationEngine:
    """
    Generate station recommendations for optimizing surf and fishing forecasts
    Provides actionable suggestions for improving data coverage
    """
    
    def __init__(self, phase_i_analyzer, quality_analyzer):
        """
        Initialize recommendation engine with analyzers
        """
        self.phase_i_analyzer = phase_i_analyzer
        self.quality_analyzer = quality_analyzer
    
    def analyze_multi_location_optimization(self, user_locations):
        """
        Analyze all user locations and provide comprehensive optimization recommendations
        Returns prioritized recommendations for improving overall data coverage
        """
        available_stations = self.phase_i_analyzer.get_available_stations()
        
        # Analyze coverage for each location
        location_analyses = []
        for location in user_locations:
            analysis = self.quality_analyzer.analyze_location_coverage(location, available_stations)
            location_analyses.append(analysis)
        
        # Generate comprehensive recommendations
        recommendations = {
            'priority_recommendations': [],
            'optimization_summary': {},
            'current_coverage': {},
            'improvement_potential': {}
        }
        
        # Analyze overall coverage quality
        overall_wave_quality = sum(a['quality_summary']['wave_quality'] for a in location_analyses) / len(location_analyses) if location_analyses else 0
        overall_atmospheric_quality = sum(a['quality_summary']['atmospheric_quality'] for a in location_analyses) / len(location_analyses) if location_analyses else 0
        overall_tide_quality = sum(a['quality_summary']['tide_quality'] for a in location_analyses) / len(location_analyses) if location_analyses else 0
        
        recommendations['current_coverage'] = {
            'wave_quality': overall_wave_quality,
            'atmospheric_quality': overall_atmospheric_quality, 
            'tide_quality': overall_tide_quality,
            'total_locations': len(location_analyses)
        }
        
        # Generate priority recommendations based on quality gaps
        if overall_wave_quality < 0.6:
            recommendations['priority_recommendations'].append({
                'type': 'critical',
                'category': 'wave_data',
                'message': f'Wave data quality is below recommended threshold ({overall_wave_quality:.1f}/1.0)',
                'impact': 'Surf forecasts may be less accurate due to distant wave buoys',
                'suggestion': 'Consider adding closer NDBC wave buoys to Phase I configuration'
            })
        
        if overall_atmospheric_quality < 0.5:
            recommendations['priority_recommendations'].append({
                'type': 'critical',
                'category': 'atmospheric_data',
                'message': f'Atmospheric data quality is below threshold ({overall_atmospheric_quality:.1f}/1.0)',
                'impact': 'Wind and pressure forecasts may lack local precision',
                'suggestion': 'Add closer NDBC atmospheric buoys or enable WeeWX station integration'
            })
        
        if overall_tide_quality < 0.7:
            recommendations['priority_recommendations'].append({
                'type': 'important',
                'category': 'tide_data',
                'message': f'Tide data quality could be improved ({overall_tide_quality:.1f}/1.0)',
                'impact': 'Fishing forecasts may be less precise for tide-dependent species',
                'suggestion': 'Consider adding backup CO-OPS tide stations'
            })
        
        # Check for locations with particularly poor coverage
        for analysis in location_analyses:
            location_quality = analysis['quality_summary']['overall_quality']
            if location_quality < 0.4:
                recommendations['priority_recommendations'].append({
                    'type': 'location_specific',
                    'category': 'poor_coverage',
                    'message': f'Location "{analysis["location"]}" has poor data coverage ({location_quality:.1f}/1.0)',
                    'impact': 'Forecasts for this location may be unreliable',
                    'suggestion': f'Consider relocating closer to existing stations or adding stations near {analysis["location"]}'
                })
        
        return recommendations
    
    def display_recommendations_interactive(self, recommendations):
        """
        Display recommendations to user and get their choices
        Returns list of accepted recommendations for implementation
        """
        if not recommendations['priority_recommendations']:
            print(f"\n{CORE_ICONS['status']} Station Coverage Analysis")
            print("Your Phase I station configuration provides good coverage for all locations.")
            print("No immediate improvements needed.")
            return []
        
        print(f"\n{CORE_ICONS['selection']} Station Coverage Recommendations")
        print("Based on your surf/fishing locations, we recommend these improvements:")
        print()
        
        accepted_recommendations = []
        for i, rec in enumerate(recommendations['priority_recommendations'], 1):
            icon = CORE_ICONS['warning'] if rec['type'] == 'critical' else CORE_ICONS['navigation']
            print(f"{icon} {i}. {rec['message']}")
            print(f"   Impact: {rec['impact']}")
            print(f"   Suggestion: {rec['suggestion']}")
            
            while True:
                choice = input(f"   Apply this recommendation? (y/n): ").strip().lower()
                if choice in ['y', 'yes']:
                    accepted_recommendations.append(rec)
                    print(f"   {CORE_ICONS['status']} Will implement")
                    break
                elif choice in ['n', 'no']:
                    print(f"   Skipped")
                    break
                else:
                    print(f"   {CORE_ICONS['warning']} Please enter y or n")
            print()
        
        return accepted_recommendations


class SurfFishingInstaller(ExtensionInstaller):
    """
    WeeWX Surf & Fishing Forecast Extension Installer
    Follows WeeWX 5.1 ExtensionInstaller patterns exactly
    """
    
    def __init__(self):
        super(SurfFishingInstaller, self).__init__(
            version="1.0.0-alpha",
            name="surf_fishing_forecasts",
            description="Local surf and fishing forecast system for WeeWX",
            author="Shane Burkhardt",
            author_email="your-email@domain.com",
            
            # CRITICAL: Use LIST format for services
            data_services=['user.surf_fishing.SurfFishingService'],
            
            files=[
                ('bin/user', ['bin/user/surf_fishing.py']),
                ('bin/user', ['bin/user/surf_fishing_fields.yaml'])
            ],
            
            config={
                'SurfFishingService': {
                    'enable': 'true',
                    'forecast_interval': '21600',  # 6 hours in seconds
                    'log_success': 'false',
                    'log_errors': 'true'
                }
            }
        )
        
        # Load YAML configuration data for installer use
        self.yaml_data = self._load_yaml_data()
    
    def _load_yaml_data(self):
        """Load YAML configuration data for installation"""
        
        yaml_file = 'bin/user/surf_fishing_fields.yaml'
        
        try:
            if os.path.exists(yaml_file):
                with open(yaml_file, 'r') as f:
                    return yaml.safe_load(f)
            else:
                # Default minimal configuration if YAML not found
                return {
                    'api_endpoints': {
                        'wavewatch_iii': {
                            'base_url': 'https://nomads.ncep.noaa.gov/pub/data/nccf/com/wave/prod/',
                            'grids': {
                                'us_east': {'bounds': [24, 50, -95, -65]},
                                'us_west': {'bounds': [24, 50, -130, -95]},
                                'glo_30m': {'bounds': [-90, 90, -180, 180]}
                            }
                        }
                    },
                    'fish_categories': {
                        'saltwater_inshore': {
                            'display_name': 'Saltwater Inshore',
                            'species': ['Striped Bass', 'Redfish', 'Snook'],
                            'pressure_preference': 'falling'
                        },
                        'mixed_bag': {
                            'display_name': 'Mixed Species',
                            'species': ['Various'],
                            'pressure_preference': 'stable'
                        }
                    }
                }
        except Exception as e:
            print(f"Warning: Could not load YAML data: {e}")
            return {}
    
    def configure(self, engine):
        """
        Called by WeeWX during installation
        Returns True for success, False for failure
        """
        
        try:
            print("\n" + "="*80)
            print("SURF & FISHING FORECAST EXTENSION INSTALLATION")
            print("="*80)
            print("Phase II: Local Surf & Fishing Forecast System")
            print("Magic Animal: Seahorse 🐟")
            print("-" * 80)
            
            # Interactive configuration
            configurator = SurfFishingConfigurator(engine.config_dict, self.yaml_data)
            config_dict, selected_locations = configurator.run_interactive_setup()
            
            # Extend database schema
            self._extend_database_schema(engine.config_dict, selected_locations)
            
            # Update engine configuration
            engine.config_dict.update(config_dict)
            
            print("\n" + "="*80)
            print("INSTALLATION COMPLETED SUCCESSFULLY!")
            print("="*80)
            print(f"{CORE_ICONS['status']} Files installed")
            print(f"{CORE_ICONS['status']} Service registered automatically")
            print(f"{CORE_ICONS['status']} Interactive configuration completed")
            print(f"{CORE_ICONS['status']} Database schema extended")
            print("-" * 80)
            print("NEXT STEPS:")
            print("1. Restart WeeWX to activate the extension:")
            print("   sudo systemctl restart weewx")
            print("2. Check WeeWX logs for successful startup:")
            print("   sudo tail -f /var/log/syslog | grep weewx")
            print("3. Forecasts will be generated every 6 hours")
            print("="*80)
            
            return True
            
        except Exception as e:
            print(f"\nInstallation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _extend_database_schema(self, config_dict, selected_locations):
        """
        Extend database schema for surf and fishing forecasts - reads YAML directly
        """
        
        progress = InstallationProgressManager()
        
        try:
            progress.show_step_progress("Creating Database Tables")
            
            # Read YAML directly here - no separate method needed
            database_schema = self.yaml_data.get('database_schema', {})
            
            if not database_schema:
                raise Exception(
                    f"{CORE_ICONS['warning']} No database_schema found in YAML configuration. "
                    "The surf_fishing_fields.yaml file must contain 'database_schema' section with table definitions."
                )
            
            with weewx.manager.open_manager_with_config(config_dict, 'wx_binding') as manager:
                
                # Process each table from YAML
                for table_key, table_config in database_schema.items():
                    if isinstance(table_config, dict) and 'name' in table_config and 'fields' in table_config:
                        table_name = table_config['name']
                        table_fields = table_config['fields']
                        
                        # Build field definitions
                        field_definitions = []
                        for field_name, field_type in table_fields.items():
                            # MYSQL FIX: Convert TEXT to VARCHAR(50) for constraint fields
                            if (field_name == "period_name" and 
                                field_type == "TEXT NOT NULL" and 
                                table_name == "marine_forecast_fishing_data"):
                                field_type = "VARCHAR(50) NOT NULL"
                            
                            field_definitions.append(f"{field_name} {field_type}")
                        
                        # Add table-specific constraints
                        constraints = []
                        if table_name == "marine_forecast_surf_data":
                            constraints.extend([
                                "FOREIGN KEY(spot_id) REFERENCES marine_forecast_surf_spots(id)",
                                "UNIQUE(spot_id, forecast_time)"
                            ])
                        elif table_name == "marine_forecast_fishing_data":
                            constraints.extend([
                                "FOREIGN KEY(spot_id) REFERENCES marine_forecast_fishing_spots(id)",
                                "UNIQUE(spot_id, forecast_date, period_name)"
                            ])
                        
                        # Combine fields and constraints
                        all_definitions = field_definitions + constraints
                        
                        # Add inline indexes to table definitions (Phase I pattern)
                        if table_name == "marine_forecast_surf_spots":
                            all_definitions.append("INDEX idx_surf_spots_active (active)")
                        elif table_name == "marine_forecast_fishing_spots":
                            all_definitions.append("INDEX idx_fishing_spots_active (active)")
                        elif table_name == "marine_forecast_surf_data":
                            all_definitions.append("INDEX idx_surf_forecasts_time (spot_id, forecast_time)")
                        elif table_name == "marine_forecast_fishing_data":
                            all_definitions.append("INDEX idx_fishing_forecasts_date (spot_id, forecast_date)")
                        
                        # Execute CREATE TABLE with inline indexes (Phase I pattern)
                        create_sql = f"""
                            CREATE TABLE IF NOT EXISTS {table_name} (
                                {', '.join(all_definitions)}
                            )
                        """
                        manager.connection.execute(create_sql)
                
                # Insert user locations into database (existing method)
                self._insert_user_locations(manager, selected_locations)
                
                print(f"  {CORE_ICONS['status']} Database tables and indexes created successfully")
            
            progress.complete_step("Creating Database Tables")
            
        except Exception as e:
            progress.show_error("Creating Database Tables", str(e))
            raise
    
    def _insert_user_locations(self, manager, selected_locations):
        """Insert user-configured locations into database"""
        
        current_time = int(time.time())
        
        # Insert surf spots
        for spot in selected_locations.get('surf_spots', []):
            manager.connection.execute("""
                INSERT OR REPLACE INTO marine_forecast_surf_spots 
                (name, latitude, longitude, bottom_type, exposure, created_date, active)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                spot['name'],
                spot['latitude'],
                spot['longitude'],
                spot['bottom_type'],
                spot['exposure'],
                current_time,
                1
            ))
        
        # Insert fishing spots
        for spot in selected_locations.get('fishing_spots', []):
            manager.connection.execute("""
                INSERT OR REPLACE INTO marine_forecast_fishing_spots 
                (name, latitude, longitude, location_type, target_category, created_date, active)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                spot['name'],
                spot['latitude'],
                spot['longitude'],
                spot['location_type'],
                spot['target_category'],
                current_time,
                1
            ))
        
        # Commit the changes
        manager.connection.commit()
        
        surf_count = len(selected_locations.get('surf_spots', []))
        fishing_count = len(selected_locations.get('fishing_spots', []))
        
        print(f"  {CORE_ICONS['status']} Inserted {surf_count} surf spots and {fishing_count} fishing spots")


        """
        Get table schemas from YAML configuration data.
        Returns dictionary of table_name -> field_definitions
        """
        
        # Get table schemas from YAML data
        table_schemas = self.yaml_data.get('table_schemas', {})
        
        required_tables = {}
        
        # Process each table definition from YAML
        for table_key, table_config in table_schemas.items():
            if isinstance(table_config, dict) and 'name' in table_config and 'fields' in table_config:
                table_name = table_config['name']
                table_fields = table_config['fields']
                
                # Convert YAML field definitions to SQL format
                if isinstance(table_fields, dict):
                    required_tables[table_name] = table_fields
        
        # Validate that required tables are found in YAML
        if not required_tables:
            raise Exception(
                f"{CORE_ICONS['warning']} No table schemas found in YAML configuration. "
                "The surf_fishing_fields.yaml file must contain 'table_schemas' section with table definitions."
            )
        
        return required_tables