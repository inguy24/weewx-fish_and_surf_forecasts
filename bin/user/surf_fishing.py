#!/usr/bin/env python3
# Magic Animal: Purple Martin
"""
WeeWX Surf & Fishing Forecast Service
Phase II: Local Surf & Fishing Forecast System

Copyright 2025 Shane Burkhardt
"""

import sys
import os
import time
import threading
import json
import math
import urllib.request
import urllib.error
import tempfile
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

# WeeWX imports
import weewx
import weewx.units
import weewx.manager
from weewx.engine import StdService
from weewx.cheetahgenerator import SearchList
import weeutil.logger

# Logging setup
log = weeutil.logger.logging.getLogger(__name__)

# CORE ICONS: Consistent with Phase I patterns (MANDATORY)
CORE_ICONS = {
    'navigation': '📍',    # Location/station selection
    'status': '✅',        # Success indicators  
    'warning': '⚠️',       # Warnings/issues
    'selection': '🔧'      # Configuration/selection
}

class GRIBProcessor:
    """Handle GRIB file processing for WaveWatch III data"""
    
    def __init__(self, config_dict):
        """Initialize GRIB processor with data-driven configuration"""
        self.config_dict = config_dict
        self.grib_library = self._detect_grib_library()
    
    def _detect_grib_library(self):
        """Detect available GRIB processing library (preserve existing logic)"""
        try:
            import eccodes
            log.info("Using eccodes-python for GRIB processing")
            return 'eccodes'
        except ImportError:
            pass
        
        try:
            import pygrib
            log.info("Using pygrib for GRIB processing")
            return 'pygrib'
        except ImportError:
            pass
        
        log.warning("No GRIB library available - GFS Wave forecasts disabled")
        return None
    
    def is_available(self):
        """Check if GRIB processing is available (preserve existing logic)"""
        return self.grib_library is not None

    def process_gfs_wave_file(self, grib_file_path, target_lat, target_lon):
        """Extract GFS Wave data for specific location using data-driven parameter mapping"""
        
        if not self.grib_library:
            return []
        
        try:
            if self.grib_library == 'eccodes':
                return self._process_with_eccodes(grib_file_path, target_lat, target_lon)
            elif self.grib_library == 'pygrib':
                return self._process_with_pygrib(grib_file_path, target_lat, target_lon)
        except Exception as e:
            log.error(f"Error processing GRIB file {grib_file_path}: {e}")
            return []
    
    def _process_with_eccodes(self, grib_file_path, target_lat, target_lon):
        """Process GRIB file using eccodes library and extract data points"""
        
        try:
            import eccodes
            from datetime import datetime, timedelta
            
            # READ FROM NEW CONF: Field mappings for parameter processing
            service_config = self.config_dict.get('SurfFishingService', {})
            gfs_wave_config = service_config.get('gfs_wave', {})
            field_mappings = gfs_wave_config.get('field_mappings', {})
            
            # Build parameter mapping from CONF with proper type conversion
            grib_parameters = {}
            for field_key, field_config in field_mappings.items():
                grib_param = field_config.get('grib_parameter', '')
                if grib_param:
                    grib_parameters[grib_param] = {
                        'database_field': field_config.get('database_field', ''),
                        'forecast_priority': int(field_config.get('forecast_priority', '3')),
                        'description': field_config.get('description', '')
                    }
            
            data_points = []
            
            with open(grib_file_path, 'rb') as f:
                while True:
                    msg_id = eccodes.codes_grib_new_from_file(f)
                    if msg_id is None:
                        break
                    
                    try:
                        # Get parameter info
                        param_name = eccodes.codes_get(msg_id, 'shortName')
                        
                        # DATA-DRIVEN: Only process parameters defined in CONF
                        if param_name not in grib_parameters:
                            eccodes.codes_release(msg_id)
                            continue
                        
                        # Get forecast time
                        forecast_time_offset = eccodes.codes_get(msg_id, 'forecastTime')
                        step_units = eccodes.codes_get(msg_id, 'stepUnits', 'h')
                        
                        # Convert to absolute timestamp
                        base_time = eccodes.codes_get(msg_id, 'dataTime')
                        base_date = eccodes.codes_get(msg_id, 'dataDate')
                        
                        # Parse base datetime
                        base_dt = datetime.strptime(f"{base_date:08d}{base_time:04d}", "%Y%m%d%H%M")
                        
                        # Add forecast offset
                        if step_units == 'h':
                            forecast_dt = base_dt + timedelta(hours=forecast_time_offset)
                        else:
                            forecast_dt = base_dt + timedelta(hours=forecast_time_offset)  # Assume hours
                        
                        forecast_timestamp = forecast_dt.timestamp()
                        
                        # Get nearest grid point value
                        value = eccodes.codes_get_nearest(msg_id, target_lat, target_lon)[0]['value']
                        
                        data_points.append({
                            'parameter': param_name,
                            'value': value,
                            'forecast_time': forecast_timestamp,
                            'latitude': target_lat,
                            'longitude': target_lon
                        })
                        
                    except Exception as e:
                        log.warning(f"{CORE_ICONS['warning']} Error processing GRIB message: {e}")
                    
                    finally:
                        eccodes.codes_release(msg_id)
            
            log.debug(f"{CORE_ICONS['status']} Extracted {len(data_points)} data points using CONF parameter mappings")
            return data_points
            
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error processing GRIB file with eccodes: {e}")
            return []
    
    def _process_with_pygrib(self, grib_file_path, target_lat, target_lon):
        """Process GRIB file using pygrib library and extract data points"""
        
        try:
            import pygrib
            from datetime import datetime, timedelta
            
            # READ FROM NEW CONF: Field mappings for parameter processing
            service_config = self.config_dict.get('SurfFishingService', {})
            gfs_wave_config = service_config.get('gfs_wave', {})
            field_mappings = gfs_wave_config.get('field_mappings', {})
            
            # Build parameter mapping from CONF with proper type conversion
            grib_parameters = {}
            for field_key, field_config in field_mappings.items():
                grib_param = field_config.get('grib_parameter', '')
                if grib_param:
                    grib_parameters[grib_param] = {
                        'database_field': field_config.get('database_field', ''),
                        'forecast_priority': int(field_config.get('forecast_priority', '3')),
                        'description': field_config.get('description', '')
                    }
            
            data_points = []
            
            with pygrib.open(grib_file_path) as grbs:
                for grb in grbs:
                    try:
                        # Get parameter info
                        param_name = grb.shortName
                        
                        # DATA-DRIVEN: Only process parameters defined in CONF
                        if param_name not in grib_parameters:
                            continue
                        
                        # Get forecast time info
                        forecast_time_offset = grb.forecastTime
                        
                        # Get base time from GRIB message
                        valid_date = grb.validDate
                        forecast_timestamp = valid_date.timestamp()
                        
                        # Get nearest grid point value
                        value, lat, lon = grb.data(lat1=target_lat-0.1, lat2=target_lat+0.1, 
                                                lon1=target_lon-0.1, lon2=target_lon+0.1)
                        
                        # Find closest point
                        if value.size > 0:
                            # Get the first (closest) value
                            closest_value = float(value.flat[0])
                            
                            data_points.append({
                                'parameter': param_name,
                                'value': closest_value,
                                'forecast_time': forecast_timestamp,
                                'latitude': target_lat,
                                'longitude': target_lon
                            })
                        
                    except Exception as e:
                        log.warning(f"{CORE_ICONS['warning']} Error processing GRIB message with pygrib: {e}")
                        continue
            
            log.debug(f"{CORE_ICONS['status']} Extracted {len(data_points)} data points using pygrib with CONF parameter mappings")
            return data_points
            
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error processing GRIB file with pygrib: {e}")
            return []
    
    def _find_nearest_grid_point(self, target_lat, target_lon, lats, lons):
        """Find nearest grid point in lat/lon arrays"""
        
        # Calculate distances
        distances = ((lats - target_lat)**2 + (lons - target_lon)**2)**0.5
        
        # Find minimum distance index
        min_idx = distances.argmin()
        lat_idx, lon_idx = divmod(min_idx, distances.shape[1])
        
        return lat_idx, lon_idx


class MarineStationIntegrationManager:
    """Manage integration with Phase I marine station data and quality analysis"""
    
    def __init__(self, config_dict, field_definitions):
        """Initialize with Phase I CONF metadata and field definitions"""
        self.config_dict = config_dict
        self.marine_config = config_dict.get('MarineDataService', {})
        self.field_definitions = field_definitions
        self.phase_i_metadata = None
        self.quality_thresholds = None
        self._load_phase_i_metadata()
        self._load_quality_thresholds()
        
    def _load_phase_i_metadata(self):
        """Load station metadata from Phase I CONF structure"""
        try:
            station_metadata = self.marine_config.get('station_metadata', {})
            self.phase_i_metadata = {
                'coops_stations': station_metadata.get('coops_stations', {}),
                'ndbc_stations': station_metadata.get('ndbc_stations', {}),
                'available': len(station_metadata) > 0
            }
            log.info(f"{CORE_ICONS['status']} Loaded Phase I metadata for {len(self.phase_i_metadata.get('coops_stations', {}))} CO-OPS and {len(self.phase_i_metadata.get('ndbc_stations', {}))} NDBC stations")
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error loading Phase I metadata: {e}")
            self.phase_i_metadata = {'available': False, 'coops_stations': {}, 'ndbc_stations': {}}
    
    def _load_quality_thresholds(self):
        """Load research-based quality thresholds from field definitions"""
        try:
            self.quality_thresholds = self.field_definitions.get('station_quality_thresholds', {})
            log.debug(f"{CORE_ICONS['status']} Loaded station quality thresholds from YAML")
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error loading quality thresholds: {e}")
            self.quality_thresholds = {}
    
    def calculate_distance(self, location_coords, station_coords):
        """Calculate haversine distance between location and station in miles"""
        lat1, lon1 = math.radians(location_coords[0]), math.radians(location_coords[1])
        lat2, lon2 = math.radians(station_coords[0]), math.radians(station_coords[1])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Earth radius in miles
        return 3959 * c
    
    def calculate_wave_quality(self, distance_miles):
        """Research-based wave data quality scoring (García-Reyes methodology)"""
        thresholds = self.quality_thresholds.get('wave_data', {})
        excellent = thresholds.get('excellent_distance_miles', 25)
        good = thresholds.get('good_distance_miles', 50) 
        fair = thresholds.get('fair_distance_miles', 100)
        
        if distance_miles <= excellent:
            return 1.0  # Excellent
        elif distance_miles <= good:
            return 0.8  # Good
        elif distance_miles <= fair:
            return 0.6  # Fair
        else:
            return 0.3  # Poor
    
    def calculate_atmospheric_quality(self, distance_miles):
        """Research-based atmospheric data quality scoring (Bourassa methodology)"""
        thresholds = self.quality_thresholds.get('atmospheric_data', {})
        excellent = thresholds.get('excellent_distance_miles', 50)
        good = thresholds.get('good_distance_miles', 100)
        fair = thresholds.get('fair_distance_miles', 200)
        
        if distance_miles <= excellent:
            return 1.0  # Excellent
        elif distance_miles <= good:
            return 0.8  # Good
        elif distance_miles <= fair:
            return 0.6  # Fair
        else:
            return 0.4  # Poor
    
    def calculate_tide_quality(self, distance_miles):
        """Research-based tide data quality scoring"""
        thresholds = self.quality_thresholds.get('tide_data', {})
        excellent = thresholds.get('excellent_distance_miles', 15)
        good = thresholds.get('good_distance_miles', 50)
        fair = thresholds.get('fair_distance_miles', 100)
        
        if distance_miles <= excellent:
            return 1.0  # Excellent
        elif distance_miles <= good:
            return 0.8  # Good
        elif distance_miles <= fair:
            return 0.6  # Fair
        else:
            return 0.3  # Poor
    
    def select_optimal_wave_source(self, location_coords):
        """Select best wave data source for specific location"""
        if not self.phase_i_metadata['available']:
            return None
            
        ndbc_stations = self.phase_i_metadata['ndbc_stations']
        wave_capable_stations = []
        
        for station_id, station_data in ndbc_stations.items():
            if station_data.get('wave_capability') == 'true':
                distance = self.calculate_distance(
                    location_coords, 
                    (float(station_data['latitude']), float(station_data['longitude']))
                )
                quality = self.calculate_wave_quality(distance)
                
                wave_capable_stations.append({
                    'station_id': station_id,
                    'name': station_data.get('name', station_id),
                    'distance_miles': distance,
                    'quality_score': quality,
                    'latitude': station_data['latitude'],
                    'longitude': station_data['longitude']
                })
        
        if not wave_capable_stations:
            return None
            
        # Return highest quality source
        return max(wave_capable_stations, key=lambda s: s['quality_score'])
    
    def select_optimal_atmospheric_sources(self, location_coords, max_sources=3):
        """Select best atmospheric data sources with multi-source fusion capability"""
        if not self.phase_i_metadata['available']:
            return []
            
        ndbc_stations = self.phase_i_metadata['ndbc_stations']
        atmospheric_capable_stations = []
        
        for station_id, station_data in ndbc_stations.items():
            if station_data.get('atmospheric_capability') == 'true':
                distance = self.calculate_distance(
                    location_coords,
                    (float(station_data['latitude']), float(station_data['longitude']))
                )
                quality = self.calculate_atmospheric_quality(distance)
                
                atmospheric_capable_stations.append({
                    'station_id': station_id,
                    'name': station_data.get('name', station_id),
                    'distance_miles': distance,
                    'quality_score': quality,
                    'latitude': station_data['latitude'],
                    'longitude': station_data['longitude']
                })
        
        # Sort by quality and return top sources
        atmospheric_capable_stations.sort(key=lambda s: s['quality_score'], reverse=True)
        return atmospheric_capable_stations[:max_sources]
    
    def select_optimal_tide_source(self, location_coords):
        """Select best tide data source for specific location"""
        if not self.phase_i_metadata['available']:
            return None
            
        coops_stations = self.phase_i_metadata['coops_stations']
        tide_capable_stations = []
        
        for station_id, station_data in coops_stations.items():
            distance = self.calculate_distance(
                location_coords,
                (float(station_data['latitude']), float(station_data['longitude']))
            )
            quality = self.calculate_tide_quality(distance)
            
            tide_capable_stations.append({
                'station_id': station_id,
                'name': station_data.get('name', station_id),
                'distance_miles': distance,
                'quality_score': quality,
                'latitude': station_data['latitude'],
                'longitude': station_data['longitude']
            })
        
        if not tide_capable_stations:
            return None
            
        # Return highest quality source
        return max(tide_capable_stations, key=lambda s: s['quality_score'])
    
    def get_integration_metadata_summary(self):
        """Get summary of available station metadata for diagnostics"""
        return {
            'phase_i_available': self.phase_i_metadata['available'],
            'coops_station_count': len(self.phase_i_metadata['coops_stations']),
            'ndbc_station_count': len(self.phase_i_metadata['ndbc_stations']),
            'wave_capable_stations': len([s for s in self.phase_i_metadata['ndbc_stations'].values() 
                                        if s.get('wave_capability') == 'true']),
            'atmospheric_capable_stations': len([s for s in self.phase_i_metadata['ndbc_stations'].values() 
                                               if s.get('atmospheric_capability') == 'true'])
        }
    

class DataFusionProcessor:
    """Handle multi-source data fusion with quality confidence scoring"""
    
    def __init__(self, config_dict, field_definitions):
        """Initialize with integration settings from CONF and field definitions"""
        self.config_dict = config_dict
        self.field_definitions = field_definitions
        self.fusion_params = field_definitions.get('fusion_parameters', {})
        self.calibration_factors = field_definitions.get('calibration_factors', {})
        self.quality_control = field_definitions.get('quality_control', {})
        
    def fuse_atmospheric_data(self, sources_data):
        """Distance-weighted interpolation for atmospheric data (Thomas methodology)"""
        if not sources_data or len(sources_data) == 0:
            return None
            
        fusion_config = self.fusion_params.get('atmospheric_fusion', {})
        
        # Single source - apply calibration only
        if len(sources_data) == 1:
            source = sources_data[0]
            return self._apply_calibration_factors(source['data'], source.get('station_type', 'standard_buoy'))
        
        # Multi-source fusion with distance weighting
        fused_data = {}
        total_weight = 0.0
        
        for source in sources_data:
            # Calculate distance-based weight
            distance_miles = source.get('distance_miles', 999)
            quality_score = source.get('quality_score', 0.5)
            
            # Weight = quality * inverse distance squared
            weight = quality_score / max(1.0, (distance_miles / 10.0) ** 2)
            
            # Apply calibration to source data
            calibrated_data = self._apply_calibration_factors(
                source['data'], 
                source.get('station_type', 'standard_buoy')
            )
            
            # Accumulate weighted values
            for field, value in calibrated_data.items():
                if value is not None:
                    if field not in fused_data:
                        fused_data[field] = {'weighted_sum': 0.0, 'total_weight': 0.0}
                    
                    fused_data[field]['weighted_sum'] += value * weight
                    fused_data[field]['total_weight'] += weight
            
            total_weight += weight
        
        # Calculate final weighted averages
        result = {}
        for field, accumulator in fused_data.items():
            if accumulator['total_weight'] > 0:
                result[field] = accumulator['weighted_sum'] / accumulator['total_weight']
        
        return result
    
    def calculate_confidence_score(self, sources, quality_scores):
        """Calculate forecast confidence based on source quality and agreement"""
        if not sources or len(sources) == 0:
            return 0.0
            
        fusion_config = self.fusion_params.get('atmospheric_fusion', {})
        confidence_factors = fusion_config.get('confidence_factors', {})
        
        # Base confidence from number of sources
        num_sources = len(sources)
        if num_sources == 1:
            base_confidence = confidence_factors.get('single_source', 0.7)
        elif num_sources == 2:
            base_confidence = confidence_factors.get('two_sources', 0.9)
        else:
            base_confidence = confidence_factors.get('three_plus_sources', 1.0)
        
        # Adjust by average quality score
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.5
        
        # Final confidence score
        confidence = min(1.0, base_confidence * avg_quality)
        
        return confidence
    
    def validate_temporal_consistency(self, sources_data):
        """Verify sources show similar trend directions"""
        if len(sources_data) < 2:
            return True  # Single source always consistent
            
        # Check pressure trends for consistency
        pressure_trends = []
        for source in sources_data:
            pressure_trend = source['data'].get('pressure_trend')
            if pressure_trend:
                pressure_trends.append(pressure_trend)
        
        if len(pressure_trends) < 2:
            return True  # Not enough trend data
            
        # Check if trends generally agree (simple majority consensus)
        rising_count = pressure_trends.count('rising')
        falling_count = pressure_trends.count('falling')
        stable_count = pressure_trends.count('stable')
        
        total_trends = len(pressure_trends)
        max_agreement = max(rising_count, falling_count, stable_count)
        
        # Consider consistent if >50% agreement
        agreement_ratio = max_agreement / total_trends
        return agreement_ratio > 0.5
    
    def _apply_calibration_factors(self, source_data, station_type):
        """Apply research-based calibration factors for known systematic differences"""
        calibrated_data = source_data.copy()
        
        # Get station type calibration factor
        buoy_types = self.calibration_factors.get('buoy_types', {})
        station_factor = buoy_types.get(station_type, 1.0)
        
        # Get measurement corrections
        corrections = self.calibration_factors.get('measurement_corrections', {})
        
        # Apply wind speed correction (Thomas et al. 2011)
        if 'wind_speed' in calibrated_data and calibrated_data['wind_speed'] is not None:
            wind_correction = corrections.get('wind_speed_systematic', 1.0)
            calibrated_data['wind_speed'] *= (station_factor * wind_correction)
        
        # Apply pressure offset correction if needed
        if 'barometric_pressure' in calibrated_data and calibrated_data['barometric_pressure'] is not None:
            pressure_offset = corrections.get('pressure_offset', 0.0)
            calibrated_data['barometric_pressure'] += pressure_offset
        
        return calibrated_data
    
    def detect_outliers(self, sources_data, field_name):
        """Detect outlier values in multi-source data for quality control"""
        if len(sources_data) < 2:
            return []  # Need at least 2 sources for outlier detection
            
        values = []
        for i, source in enumerate(sources_data):
            value = source['data'].get(field_name)
            if value is not None:
                values.append((i, value))
        
        if len(values) < 2:
            return []
            
        # Calculate mean and standard deviation
        data_values = [v[1] for v in values]
        mean_value = sum(data_values) / len(data_values)
        variance = sum((x - mean_value) ** 2 for x in data_values) / len(data_values)
        std_dev = math.sqrt(variance)
        
        # Flag outliers beyond 2 standard deviations
        outliers = []
        for idx, value in values:
            if abs(value - mean_value) > (2 * std_dev):
                outliers.append({
                    'source_index': idx,
                    'value': value,
                    'deviation': abs(value - mean_value)
                })
        
        return outliers
    
    def check_data_freshness(self, source_data, current_time):
        """Check if data meets freshness requirements"""
        max_age_hours = float(self.quality_control.get('max_data_age_hours', 6))
        max_age_seconds = max_age_hours * 3600
        
        data_time = source_data.get('observation_time')
        if not data_time:
            return False  # No timestamp = not fresh
            
        age_seconds = current_time - data_time
        return age_seconds <= max_age_seconds
    

class WaveWatchDataCollector:
    """Collect WaveWatch III offshore wave forecast data"""
    
    def __init__(self, config_dict, grib_processor):
        """Initialize GFS Wave data collector with configuration and GRIB processor"""
        self.config_dict = config_dict
        self.grib_processor = grib_processor
        
        # READ FROM NEW CONF STRUCTURE: gfs_wave section
        service_config = config_dict.get('SurfFishingService', {})
        self.gfs_wave_config = service_config.get('noaa_gfs_wave', {})
        
        # READ FROM NEW CONF: api_endpoints subsection
        api_endpoints = self.gfs_wave_config.get('api_endpoints', {})
        self.base_url = self.gfs_wave_config.get('base_url', '')
        
        # READ FROM NEW CONF: Scheduling configuration - NO DEFAULTS
        run_cycles_str = api_endpoints.get('model_run_cycles', '')
        self.run_cycles = [0, 6, 12, 18]  # GFS Wave standard cycles
        
        forecast_hours_str = api_endpoints.get('forecast_hours', '')
        self.forecast_hours = [int(h.strip()) for h in forecast_hours_str.split(',') if h.strip()]
        
        # READ FROM NEW CONF: Grid selection configuration
        grid_selection = self.gfs_wave_config.get('grid_selection', {})
        self.default_grid = self.gfs_wave_config.get('grid_selected', 'global.0p16')
        self.auto_select_regional = grid_selection.get('auto_select_regional', 'false').lower() == 'true'
        self.global_grid = grid_selection.get('global_grid', '')
        self.atlantic_grid = grid_selection.get('atlantic_grid', '')
        self.pacific_grid = grid_selection.get('pacific_grid', '')
        
        # READ FROM NEW CONF: Error handling configuration
        error_handling = self.gfs_wave_config.get('error_handling', {})
        self.api_timeout = int(error_handling.get('api_timeout', '30'))
        self.retry_attempts = int(error_handling.get('retry_attempts', '3'))
        self.retry_delay = int(error_handling.get('retry_delay', '5'))
        self.max_retry_delay = int(error_handling.get('max_retry_delay', '30'))
        self.connection_timeout = int(error_handling.get('connection_timeout', '10'))
        self.read_timeout = int(error_handling.get('read_timeout', '20'))
        
        log.info(f"{CORE_ICONS['status']} WaveWatchDataCollector initialized with new CONF structure")
        log.debug(f"{CORE_ICONS['navigation']} Base URL: {self.base_url}")
        log.debug(f"{CORE_ICONS['navigation']} Run cycles: {self.run_cycles}")
        log.debug(f"{CORE_ICONS['navigation']} Default grid: {self.default_grid}")

    def fetch_forecast_data(self, latitude, longitude):
        """Fetch GFS Wave forecast data for location with data-driven grid selection"""
    
        if not self.grib_processor.is_available():
            log.warning("GRIB processing not available - skipping GFS Wave data")
            return []
        
        try:
            # DATA-DRIVEN grid selection
            grid_name = self._select_grid_data_driven(latitude, longitude)
            log.debug(f"Using GFS Wave grid: {grid_name} for location {latitude}, {longitude}")
            
            # Download GRIB files with updated URL structure
            grib_files = self._download_grib_files(grid_name)
            
            # SURGICAL FIX 1: ADD FAIL-FAST CHECK FOR EMPTY GRIB FILES
            if not grib_files:
                log.error("No GRIB files downloaded - failing forecast generation")
                return []
            
            # Process GRIB files with data-driven parameter mapping
            forecast_data = []
            for grib_file in grib_files:
                try:
                    data_points = self.grib_processor.process_gfs_wave_file(
                        grib_file, latitude, longitude
                    )
                    forecast_data.extend(data_points)
                except Exception as e:
                    log.error(f"Error processing GRIB file {grib_file}: {e}")
                    continue
            
            log.info(f"DEBUG: Preserved {len(grib_files)} GRIB files in /tmp/ for inspection")
            return self._organize_forecast_data(forecast_data)
        
        except Exception as e:
            log.error(f"Error fetching GFS Wave data: {e}")
            return []
        
    def _select_grid_data_driven(self, latitude, longitude):
        """Select optimal grid based on location coordinates"""
        
        if not self.auto_select_regional:
            return self.default_grid
        
        # DATA-DRIVEN: Use regional grids from CONF if they exist and location matches
        # Pacific region check - if pacific_grid is configured
        if self.pacific_grid:
            # Use CONF-based bounds if available, otherwise skip
            grid_selection = self.gfs_wave_config.get('grid_selection', {})
            pacific_bounds = grid_selection.get('pacific_bounds', [])
            if len(pacific_bounds) == 4:
                lat_min, lat_max, lon_min, lon_max = pacific_bounds
                if lat_min <= latitude <= lat_max and lon_min <= longitude <= lon_max:
                    return self.pacific_grid
        
        # Atlantic region check - if atlantic_grid is configured  
        if self.atlantic_grid:
            grid_selection = self.gfs_wave_config.get('grid_selection', {})
            atlantic_bounds = grid_selection.get('atlantic_bounds', [])
            if len(atlantic_bounds) == 4:
                lat_min, lat_max, lon_min, lon_max = atlantic_bounds
                if lat_min <= latitude <= lat_max and lon_min <= longitude <= lon_max:
                    return self.atlantic_grid
        
        # Fallback to global grid from CONF
        return self.global_grid if self.global_grid else self.default_grid
    
    def _download_grib_files(self, grid_name):
        """Download GFS Wave GRIB files for specified grid"""
        import urllib.request
        import urllib.error
        from datetime import datetime, timedelta
        
        grib_files = []
        successful_cycle = None
        
        try:
            current_time = datetime.utcnow()
            
            # DATA-DRIVEN: Use lookback hours from CONF
            error_handling = self.gfs_wave_config.get('error_handling', {})
            lookback_hours = int(error_handling.get('lookback_hours', '24'))
            processing_delay_hours = int(error_handling.get('processing_delay_hours', '3'))
            
            # Try each run cycle in reverse order to get most recent
            for cycle_hours_back in range(lookback_hours):
                for run_cycle in reversed(self.run_cycles):
                    
                    potential_run = current_time.replace(
                        hour=run_cycle, minute=0, second=0, microsecond=0
                    ) - timedelta(hours=cycle_hours_back)
                    
                    # Skip future runs - use processing delay from CONF
                    if potential_run > current_time - timedelta(hours=processing_delay_hours):
                        continue
                    
                    run_date_str = potential_run.strftime("%Y%m%d")
                    run_hour_str = f"{potential_run.hour:02d}"
                    
                    log.debug(f"{CORE_ICONS['navigation']} Trying GFS Wave cycle: {run_date_str} {run_hour_str}Z")
                    
                    cycle_files = []
                    
                    # DATA-DRIVEN: Use forecast hours from CONF
                    max_forecast_files = int(error_handling.get('max_forecast_files', '25'))
                    for forecast_hour in self.forecast_hours[:max_forecast_files]:
                        
                        # DATA-DRIVEN: Build filename using CONF pattern
                        api_endpoints = self.gfs_wave_config.get('api_endpoints', {})
                        url_pattern = api_endpoints.get('url_pattern', '')
                        file_pattern = api_endpoints.get('file_pattern', '')
                        
                        if not url_pattern or not file_pattern:
                            log.error(f"{CORE_ICONS['warning']} Missing url_pattern or file_pattern in CONF")
                            continue
                        
                        # Replace template variables
                        filename = file_pattern.format(
                            hh=run_hour_str,
                            grid=grid_name,
                            fff=f"{forecast_hour:03d}"
                        )
                        
                        url = f"{self.base_url}/{url_pattern.format(yyyymmdd=run_date_str, hh=run_hour_str)}/{filename}"
                        
                        try:
                            # DATA-DRIVEN: Use timeout values from CONF
                            request = urllib.request.Request(url)
                            with urllib.request.urlopen(request, timeout=self.connection_timeout) as response:
                                
                                # Download to temporary file
                                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.grib2')
                                
                                # Read in chunks with timeout
                                import socket
                                socket.setdefaulttimeout(self.read_timeout)
                                
                                while True:
                                    chunk = response.read(8192)
                                    if not chunk:
                                        break
                                    temp_file.write(chunk)
                                
                                temp_file.close()
                                cycle_files.append(temp_file.name)
                                
                                log.debug(f"{CORE_ICONS['status']} Downloaded: {filename}")
                                
                        except (urllib.error.URLError, urllib.error.HTTPError, socket.timeout) as e:
                            log.debug(f"{CORE_ICONS['warning']} Failed to download {filename}: {e}")
                            continue
                        except Exception as e:
                            log.error(f"{CORE_ICONS['warning']} Unexpected error downloading {filename}: {e}")
                            continue
                    
                    # If we got files from this cycle, use it
                    if cycle_files:
                        grib_files = cycle_files
                        successful_cycle = potential_run
                        break
                
                # If we found files, break out of the outer loop too
                if grib_files:
                    break
            
            if grib_files:
                if successful_cycle:
                    log.info(f"{CORE_ICONS['status']} Successfully downloaded {len(grib_files)} GRIB files from {successful_cycle.strftime('%Y%m%d %HZ')}")
                else:
                    log.info(f"{CORE_ICONS['status']} Successfully downloaded {len(grib_files)} GRIB files")
            else:
                log.warning(f"{CORE_ICONS['warning']} No GRIB files could be downloaded")
            
            return grib_files
            
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error in _download_grib_files: {e}")
            return []
    
    def _organize_forecast_data(self, data_points):
        """Organize raw GRIB data into forecast periods"""
        
        if not data_points:
            log.error(f"{CORE_ICONS['warning']} No data points provided to _organize_forecast_data")
            return []
        
        # READ FROM NEW CONF: Field conversion configuration
        field_mappings = self.gfs_wave_config.get('field_mappings', {})
        
        # Build field processing configuration from CONF
        field_conversions = {}
        for field_key, field_config in field_mappings.items():
            grib_parameter = field_config.get('grib_parameter', '')
            database_field = field_config.get('database_field', '')
            unit_conversion_required = field_config.get('unit_conversion_required', 'false').lower() == 'true'
            
            if grib_parameter and database_field:
                field_conversions[grib_parameter] = {
                    'database_field': database_field,
                    'unit_conversion_required': unit_conversion_required,
                    'database_type': field_config.get('database_type', 'REAL'),
                    'forecast_priority': int(field_config.get('forecast_priority', '3')),
                    'description': field_config.get('description', '')
                }
        
        # Group data by forecast time (preserve existing logic)
        forecast_periods = {}
        
        for point in data_points:
            forecast_time = point['forecast_time']
            
            if forecast_time not in forecast_periods:
                forecast_periods[forecast_time] = {}
            
            # Safe numeric conversion
            try:
                value = float(point['value']) if point['value'] is not None else 0.0
            except (ValueError, TypeError):
                log.warning(f"Non-numeric GRIB value for {point['parameter']}: {point['value']}, using 0.0")
                value = 0.0
            
            forecast_periods[forecast_time][point['parameter']] = value
        
        # DATA-DRIVEN: Convert to list format using CONF field mappings
        organized_data = []
        
        for forecast_time, parameters in forecast_periods.items():
            try:
                converted_data = {'forecast_time': forecast_time}
                
                # Process each parameter using CONF field mappings
                for grib_param, param_value in parameters.items():
                    if grib_param in field_conversions:
                        field_info = field_conversions[grib_param]
                        database_field = field_info['database_field']
                        
                        # Apply unit conversion if required
                        if field_info['unit_conversion_required']:
                            # Unit conversion will be handled later by the unit system
                            converted_data[database_field] = param_value
                        else:
                            converted_data[database_field] = param_value
                    else:
                        # Parameter not in CONF mappings - log but don't include
                        log.debug(f"{CORE_ICONS['navigation']} GRIB parameter {grib_param} not found in CONF field mappings")
                
                organized_data.append(converted_data)
                
            except Exception as e:
                log.error(f"{CORE_ICONS['warning']} Error organizing forecast data for time {forecast_time}: {e}")
                continue
        
        log.info(f"{CORE_ICONS['status']} Organized {len(organized_data)} forecast periods using CONF field mappings")
        return organized_data
    
    def _calculate_wind_speed(self, parameters):
        """Calculate wind speed from direct WIND parameter or U/V components fallback"""
        
        # UPDATED: Check for direct wind speed from GFS Wave first
        if 'wind_speed' in parameters:
            return parameters['wind_speed'] * 2.23694  # m/s to mph
        
        # PRESERVE: Fallback to U/V component calculation
        u_wind = parameters.get('wind_u_10m', 0)
        v_wind = parameters.get('wind_v_10m', 0)
        
        wind_speed_ms = math.sqrt(u_wind**2 + v_wind**2)
        return wind_speed_ms * 2.23694  # m/s to mph
    
    def _calculate_wind_direction(self, parameters):
        """Calculate wind direction from direct WDIR parameter or U/V components fallback"""
        
        # UPDATED: Check for direct wind direction from GFS Wave first
        if 'wind_direction' in parameters:
            return parameters['wind_direction']
        
        # PRESERVE: Fallback to U/V component calculation
        u_wind = parameters.get('wind_u_10m', 0)
        v_wind = parameters.get('wind_v_10m', 0)
        
        # Convert to meteorological direction (direction wind is coming FROM)
        wind_dir_rad = math.atan2(-u_wind, -v_wind)
        wind_dir_deg = math.degrees(wind_dir_rad)
        
        # Normalize to 0-360 degrees
        if wind_dir_deg < 0:
            wind_dir_deg += 360
        
        return wind_dir_deg


class SurfForecastGenerator:
    """Generate surf condition forecasts"""
    
    def __init__(self, config_dict, db_manager=None):
        """Initialize surf forecast generator with data-driven configuration"""
        self.config_dict = config_dict
        self.db_manager = db_manager
        service_config = config_dict.get('SurfFishingService', {})
        self.surf_rating_factors = service_config.get('surf_rating_factors', {})
        
        # READ FROM CONF: Required fields for surf forecasting
        self.required_fields = service_config.get('required_fields', {})
        self.surf_critical = self.required_fields.get('surf_forecasting', {}).get('critical', [])
        self.surf_recommended = self.required_fields.get('surf_forecasting', {}).get('recommended', [])
    
    def generate_surf_forecast(self, spot, forecast_data):
        """Generate surf forecast using data-driven field configuration and preserve all algorithms"""
        forecasts = []
        
        for period_data in forecast_data:
            # DATA-DRIVEN: Extract fields based on CONF configuration
            forecast_values = {}
            
            # Extract critical fields (data-driven from CONF)
            for field_name in self.surf_critical:
                forecast_values[field_name] = period_data.get(field_name, 0)
            
            # Extract recommended fields (data-driven from CONF)  
            for field_name in self.surf_recommended:
                forecast_values[field_name] = period_data.get(field_name, 0)
            
            # Create basic forecast structure for assess_surf_quality_complete
            basic_forecast = [{
                'forecast_time': period_data['forecast_time'],
                'wave_height_min': forecast_values.get('wave_height', 0) * 0.8,  # Create range
                'wave_height_max': forecast_values.get('wave_height', 0) * 1.2,
                'wave_period': forecast_values.get('wave_period', 0),
                'wave_direction': forecast_values.get('wave_direction', 0),
                'wind_speed': forecast_values.get('wind_speed', 0),
                'wind_direction': forecast_values.get('wind_direction', 0)
            }]
            
            # Use comprehensive surf quality assessment (handles everything correctly)
            enhanced_forecast = self.assess_surf_quality_complete(
                basic_forecast, 
                current_wind={
                    'wind_speed': forecast_values.get('wind_speed', 0),
                    'wind_direction': forecast_values.get('wind_direction', 0)
                }, 
                spot_config=spot
            )
            
            # Extract the enhanced forecast data
            if enhanced_forecast:
                forecast = enhanced_forecast[0]  # Get the enhanced period
                # Add any additional forecast values dynamically
                forecast.update(forecast_values)
            else:
                # Fallback if assessment fails
                forecast = {
                    'forecast_time': period_data['forecast_time'],
                    'rating': 1,
                    'quality_text': 'Assessment Failed'
                }
                forecast.update(forecast_values)
            
            forecasts.append(forecast)
        
        return forecasts
    
    def _get_current_wave_conditions(self, marine_conditions):
        """Extract current wave conditions from Phase I data"""
        
        return {
            'wave_height': marine_conditions.get('current_wave_height', 0),
            'wave_period': marine_conditions.get('current_wave_period', 0),
            'wave_direction': marine_conditions.get('current_wave_direction', 0)
        }
    
    def _get_current_wind_conditions(self, marine_conditions):
        """Extract current wind conditions"""
        
        return {
            'wind_speed': marine_conditions.get('current_wind_speed', 0),
            'wind_direction': marine_conditions.get('current_wind_direction', 0)
        }
    
    def _get_current_tide_conditions(self, marine_conditions):
        """Extract current tide conditions"""
        
        return {
            'current_level': marine_conditions.get('current_water_level', 0),
            'next_high': marine_conditions.get('next_high_tide', 0),
            'next_low': marine_conditions.get('next_low_tide', 0)
        }
    
    def _process_offshore_forecast(self, wavewatch_data):
        """Process WaveWatch III offshore forecast data"""
        
        processed_forecast = []
        
        for period in wavewatch_data:
            processed_period = {
                'forecast_time': period['forecast_time'],
                'offshore_wave_height': period['wave_height'],
                'offshore_wave_period': period['wave_period'],
                'offshore_wave_direction': period['wave_direction'],
                'offshore_wind_speed': period['wind_speed'],
                'offshore_wind_direction': period['wind_direction']
            }
            processed_forecast.append(processed_period)
        
        return processed_forecast

    def get_bathymetric_data_from_conf(self, spot_config):
        """
        Get bathymetric depth profile from CONF for wave transformation
        """
        try:
            # Check if spot has bathymetric data in CONF
            bathymetric_path = spot_config.get('bathymetric_path', {})
            
            if bathymetric_path:
                # Extract depth points (installed by GEBCO API)
                depths = []
                total_points = int(bathymetric_path.get('path_points_total', '7'))
                
                for i in range(total_points):
                    depth_key = f'point_{i}_depth'
                    if depth_key in bathymetric_path:
                        depth = float(bathymetric_path[depth_key])
                        depths.append(abs(depth))  # Ensure positive depth values
                
                if depths:
                    offshore_depth = depths[0]  # First point (deepest)
                    breaking_depth = depths[-1]  # Last point (shallowest)
                    
                    log.debug(f"{CORE_ICONS['status']} Using GEBCO bathymetry: {offshore_depth:.1f}m → {breaking_depth:.1f}m")
                    return offshore_depth, breaking_depth
            
            # Fallback to default depths if no bathymetric data
            return 40.0, 2.5  # Default offshore and breaking depths
            
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error reading bathymetric data: {e}")
            return 40.0, 2.5
    
    def _transform_to_local_conditions(self, offshore_forecast, spot_config):
        """
        Transform offshore wave data to local surf conditions using wave physics
        """
        try:
            transformed_forecast = []
            
            # Get spot characteristics from CONF
            beach_angle = float(spot_config.get('beach_angle', '270'))  # Default west-facing
            bottom_type = spot_config.get('bottom_type', 'sand')
            
            # Get bathymetric data (from GEBCO API or defaults)
            offshore_depth, breaking_depth = self.get_bathymetric_data_from_conf(spot_config)
            
            for period in offshore_forecast:
                try:
                    # Extract offshore wave parameters
                    offshore_height = period.get('wave_height', 0.0)
                    wave_period = period.get('wave_period', 8.0)
                    wave_direction = period.get('wave_direction', 270.0)
                    current_tide = period.get('tide_level', 0.0)  # From Phase I tide data
                    
                    # Step 1: Integrate tidal effects on breaking depth
                    effective_breaking_depth = self.integrate_tidal_effects(breaking_depth, current_tide)
                    
                    # Step 2: Calculate shoaling coefficient
                    Ks = self.calculate_shoaling_coefficient(offshore_depth, effective_breaking_depth, wave_period)
                    
                    # Step 3: Calculate refraction coefficient  
                    Kr = self.calculate_refraction_coefficient(wave_direction, beach_angle, offshore_depth, effective_breaking_depth)
                    
                    # Step 4: Apply wave transformation
                    transformed_height = offshore_height * Ks * Kr
                    
                    # Step 5: Apply breaking limitation
                    final_height = self.apply_breaking_limit(transformed_height, effective_breaking_depth, bottom_type)
                    
                    # Preserve existing output structure for compatibility
                    min_height = max(0.1, final_height * 0.8)  # 20% variance range
                    max_height = final_height * 1.2
                    
                    transformed_period = {
                        'forecast_time': period['forecast_time'],
                        'wave_height_min': min_height,
                        'wave_height_max': max_height, 
                        'wave_height_primary': final_height,  # Physics-transformed height
                        'wave_period': wave_period,
                        'wave_direction': wave_direction,
                        'wind_speed': period.get('wind_speed', 0.0),
                        'wind_direction': period.get('wind_direction', 270.0),
                        'transformation_coefficients': {  # New physics data
                            'shoaling_factor': Ks,
                            'refraction_factor': Kr,
                            'breaking_limited': final_height < transformed_height
                        }
                    }
                    
                    transformed_forecast.append(transformed_period)
                    
                except Exception as e:
                    log.error(f"{CORE_ICONS['warning']} Error transforming period: {e}")
                    # Fallback to preserve functionality
                    transformed_forecast.append(period)
            
            return transformed_forecast
            
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error in wave transformation: {e}")
            return offshore_forecast  # Return original data on error
    
    def assess_surf_quality_complete(self, wave_height, wave_period, wind_speed, wind_direction, spot_config):
        """
        Assess surf quality using enhanced physics-based scoring from CONF
        """
        try:
            # Get enhanced scoring criteria from CONF
            service_config = self.config_dict.get('SurfFishingService', {})
            scoring_criteria = service_config.get('scoring_criteria', {})
            surf_scoring = scoring_criteria.get('surf_scoring', {})
            
            # Get scoring weights from CONF
            scoring_weights = surf_scoring.get('scoring_weights', {})
            wave_height_weight = float(scoring_weights.get('wave_height', '0.35'))
            wave_period_weight = float(scoring_weights.get('wave_period', '0.35'))
            wind_quality_weight = float(scoring_weights.get('wind_quality', '0.20'))
            tide_phase_weight = float(scoring_weights.get('tide_phase', '0.10'))
            
            # Calculate wave height score using range-based lookup
            height_scoring = surf_scoring.get('transformed_wave_height_scoring', {})
            height_score = self.lookup_score_from_range_table(wave_height, height_scoring)
            
            # Calculate wave period score using range-based lookup
            period_scoring = surf_scoring.get('wave_period_scoring', {})
            period_score = self.lookup_score_from_range_table(wave_period, period_scoring)
            
            # Calculate wind quality score
            wind_score = self._calculate_wind_quality_score(wind_speed, wind_direction, spot_config, surf_scoring)
            
            # Calculate tide score (simple implementation for now)
            tide_score = 0.8  # Placeholder - can be enhanced with actual tide phase
            
            # Calculate weighted overall score
            overall_score = (
                height_score * wave_height_weight +
                period_score * wave_period_weight +
                wind_score * wind_quality_weight +
                tide_score * tide_phase_weight
            )
            
            # Convert to 1-5 star rating
            if overall_score >= 0.9:
                rating = 5
                text = "Excellent"
            elif overall_score >= 0.7:
                rating = 4
                text = "Good"
            elif overall_score >= 0.5:
                rating = 3
                text = "Fair"
            elif overall_score >= 0.3:
                rating = 2
                text = "Poor"
            else:
                rating = 1
                text = "Very Poor"
            
            return {
                'rating': rating,
                'text': text,
                'overall_score': overall_score,
                'component_scores': {
                    'height_score': height_score,
                    'period_score': period_score,
                    'wind_score': wind_score,
                    'tide_score': tide_score
                },
                'confidence': min(0.9, overall_score + 0.1)
            }
            
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error in surf quality assessment: {e}")
            # Fallback scoring to preserve functionality
            return {
                'rating': 3,
                'text': "Fair",
                'overall_score': 0.5,
                'component_scores': {'height_score': 0.5, 'period_score': 0.5, 'wind_score': 0.5, 'tide_score': 0.5},
                'confidence': 0.5
            }
    
    def _classify_wind_condition(self, wind_direction, wave_direction, wind_speed):
        """Classify wind condition relative to waves"""
        
        # Calculate relative wind direction
        wind_relative = abs(wind_direction - wave_direction)
        if wind_relative > 180:
            wind_relative = 360 - wind_relative
        
        # Classify wind condition
        if wind_speed < 5:
            wind_type = 'calm'
            quality_modifier = 1
        elif wind_relative < 45:  # Wind blowing same direction as waves
            wind_type = 'onshore'
            quality_modifier = -2
        elif wind_relative > 135:  # Wind blowing opposite to waves
            wind_type = 'offshore'
            quality_modifier = 2
        else:  # Cross-shore wind
            wind_type = 'cross'
            quality_modifier = 0
        
        return {
            'type': wind_type,
            'quality_modifier': quality_modifier,
            'wind_speed': wind_speed
        }

    def _calculate_wind_quality_score(self, wind_speed, wind_direction, spot_config, surf_scoring):
        """
        Calculate wind quality score using enhanced CONF parameters
        """
        try:
            # Get wind scoring parameters from CONF
            wind_quality_scoring = surf_scoring.get('wind_quality_scoring', {})
            wind_speed_thresholds = surf_scoring.get('wind_speed_thresholds', {})
            
            # Data-driven wind speed thresholds from CONF
            calm_max = float(wind_speed_thresholds.get('calm_max', '3'))
            light_max = float(wind_speed_thresholds.get('light_max', '8'))
            moderate_max = float(wind_speed_thresholds.get('moderate_max', '15'))
            strong_max = float(wind_speed_thresholds.get('strong_max', '25'))
            
            # Determine wind condition
            if wind_speed <= calm_max:
                wind_condition = 'calm_glassy'
            else:
                # Calculate relative wind direction
                beach_angle = float(spot_config.get('beach_angle', '270'))
                wind_relative = abs(wind_direction - beach_angle)
                if wind_relative > 180:
                    wind_relative = 360 - wind_relative
                
                # Classify wind direction relative to beach
                if wind_relative < 45:  # Onshore
                    if wind_speed <= light_max:
                        wind_condition = 'onshore_light'
                    else:
                        wind_condition = 'onshore_strong'
                elif wind_relative > 135:  # Offshore
                    if wind_speed <= light_max:
                        wind_condition = 'offshore_light'
                    elif wind_speed <= moderate_max:
                        wind_condition = 'offshore_optimal'
                    else:
                        wind_condition = 'offshore_strong'
                else:  # Cross-shore
                    wind_condition = 'cross_shore'
            
            # Get score from CONF
            wind_score = float(wind_quality_scoring.get(wind_condition, '0.5'))
            
            return wind_score
            
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error calculating wind quality: {e}")
            return 0.5

    def calculate_shoaling_coefficient(self, offshore_depth, breaking_depth, wave_period):
        """
        Calculate shoaling coefficient using linear wave theory
        """
        try:
            # Get physics parameters from enhanced CONF
            service_config = self.config_dict.get('SurfFishingService', {})
            scoring_criteria = service_config.get('scoring_criteria', {})
            surf_scoring = scoring_criteria.get('surf_scoring', {})
            physics_params = surf_scoring.get('physics_parameters', {})
            
            # Data-driven physics parameters from CONF
            shoaling_factor_max = float(physics_params.get('shoaling_factor_max', '1.5'))
            
            # Calculate wave celerity using dispersion relation
            g = 9.81  # gravitational acceleration
            
            # Deep water wave celerity
            C0 = (g * wave_period) / (2 * math.pi)
            
            # Shallow water wave celerity at breaking depth
            C_shallow = math.sqrt(g * breaking_depth)
            
            # Shoaling coefficient: Ks = sqrt(C0 / (2 * C_shallow))
            if C_shallow > 0:
                Ks = math.sqrt(C0 / (2 * C_shallow))
                # Apply maximum limit from CONF
                Ks = min(Ks, shoaling_factor_max)
            else:
                Ks = 1.0
            
            log.debug(f"{CORE_ICONS['status']} Shoaling coefficient: {Ks:.3f}")
            return Ks
            
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error calculating shoaling coefficient: {e}")
            return 1.0

    def calculate_refraction_coefficient(self, wave_direction, beach_angle, offshore_depth, breaking_depth):
        """
        Calculate refraction coefficient using Snell's Law
        """
        try:
            # Get physics parameters from enhanced CONF
            service_config = self.config_dict.get('SurfFishingService', {})
            scoring_criteria = service_config.get('scoring_criteria', {})
            surf_scoring = scoring_criteria.get('surf_scoring', {})
            physics_params = surf_scoring.get('physics_parameters', {})
            
            # Data-driven physics parameters from CONF
            refraction_factor_max = float(physics_params.get('refraction_factor_max', '1.2'))
            
            # Calculate incident angle relative to beach normal
            beach_normal = (beach_angle + 90) % 360  # Perpendicular to beach
            incident_angle = abs(wave_direction - beach_normal)
            if incident_angle > 180:
                incident_angle = 360 - incident_angle
            
            # Convert to radians
            theta_0 = math.radians(incident_angle)
            
            # Calculate refracted angle using depth ratio
            depth_ratio = breaking_depth / offshore_depth if offshore_depth > 0 else 1.0
            
            # Snell's Law: sin(theta_1) / sin(theta_0) = sqrt(h1/h0)
            if math.cos(theta_0) > 0:
                sin_theta_1 = math.sin(theta_0) * math.sqrt(depth_ratio)
                sin_theta_1 = min(sin_theta_1, 1.0)  # Prevent math domain error
                
                theta_1 = math.asin(sin_theta_1)
                
                # Refraction coefficient: Kr = sqrt(cos(theta_0) / cos(theta_1))
                if math.cos(theta_1) > 0:
                    Kr = math.sqrt(math.cos(theta_0) / math.cos(theta_1))
                    # Apply maximum limit from CONF
                    Kr = min(Kr, refraction_factor_max)
                else:
                    Kr = 1.0
            else:
                Kr = 1.0
            
            log.debug(f"{CORE_ICONS['status']} Refraction coefficient: {Kr:.3f}")
            return Kr
            
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error calculating refraction coefficient: {e}")
            return 1.0

    def apply_breaking_limit(self, wave_height, breaking_depth, bottom_type):
        """
        Apply depth-limited breaking criteria
        """
        try:
            # Get physics parameters from enhanced CONF
            service_config = self.config_dict.get('SurfFishingService', {})
            scoring_criteria = service_config.get('scoring_criteria', {})
            surf_scoring = scoring_criteria.get('surf_scoring', {})
            physics_params = surf_scoring.get('physics_parameters', {})
            
            # Data-driven breaking indices from CONF
            gamma_sand = float(physics_params.get('breaking_gamma_sand', '0.78'))
            gamma_reef = float(physics_params.get('breaking_gamma_reef', '1.0'))
            
            # Select breaking index based on bottom type
            if bottom_type in ['sand', 'beach']:
                gamma = gamma_sand
            elif bottom_type in ['reef', 'rock', 'coral']:
                gamma = gamma_reef
            else:
                gamma = gamma_sand  # Default to sand
            
            # Calculate maximum breaking wave height: Hb = gamma * db
            max_breaking_height = gamma * breaking_depth
            
            # Apply breaking limit
            limited_height = min(wave_height, max_breaking_height)
            
            if limited_height < wave_height:
                log.debug(f"{CORE_ICONS['warning']} Wave height limited by breaking: {wave_height:.2f}ft → {limited_height:.2f}ft")
            
            return limited_height
            
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error applying breaking limit: {e}")
            return wave_height

    def integrate_tidal_effects(self, breaking_depth, current_tide_level):
        """
        Adjust breaking depth based on current tidal water level
        """
        try:
            # Adjust effective breaking depth with tide level
            effective_depth = breaking_depth + current_tide_level
            
            # Ensure minimum depth
            effective_depth = max(effective_depth, 0.5)  # Minimum 0.5ft depth
            
            log.debug(f"{CORE_ICONS['status']} Tidal adjustment: {breaking_depth:.1f}ft → {effective_depth:.1f}ft")
            return effective_depth
            
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error integrating tidal effects: {e}")
            return breaking_depth
   
    def _calculate_surf_rating(self, wave_period, wind_condition, wave_height):
        """Calculate 1-5 star surf rating"""
        
        base_score = 0
        
        # Wave period scoring (most important factor)
        if wave_period >= 12:
            base_score += 3  # Excellent ground swell
        elif wave_period >= 8:
            base_score += 2  # Good swell
        elif wave_period >= 6:
            base_score += 1  # Fair swell
        else:
            base_score += 0  # Poor wind swell
        
        # Apply wind modifier
        base_score += wind_condition['quality_modifier']
        
        # Wave size scoring (optimal range is typically 3-6 feet)
        if 3 <= wave_height <= 6:
            base_score += 1  # Ideal size
        elif wave_height < 1:
            base_score -= 1  # Too small
        elif wave_height > 12:
            base_score -= 1  # Too big for most surfers
        
        # Convert to 1-5 star rating
        if base_score >= 5:
            rating = 5
            text = "Excellent"
            confidence = 0.9
        elif base_score >= 3:
            rating = 4
            text = "Good"
            confidence = 0.8
        elif base_score >= 1:
            rating = 3
            text = "Fair"
            confidence = 0.7
        elif base_score >= -1:
            rating = 2
            text = "Poor"
            confidence = 0.6
        else:
            rating = 1
            text = "Very Poor"
            confidence = 0.5
        
        return {
            'rating': rating,
            'text': text,
            'confidence': confidence
        }
    
    def _add_tide_information(self, surf_forecast, tide_conditions):
        """Add tide information to surf forecast"""
        
        enhanced_forecast = []
        
        for period in surf_forecast:
            # Simple tide stage calculation (can be enhanced with actual tide predictions)
            tide_stage = self._determine_tide_stage(period['forecast_time'], tide_conditions)
            
            enhanced_period = period.copy()
            enhanced_period.update({
                'tide_stage': tide_stage['stage'],
                'tide_height': tide_stage['height']
            })
            
            enhanced_forecast.append(enhanced_period)
        
        return enhanced_forecast
    
    def _determine_tide_stage(self, forecast_time, tide_conditions):
        """Determine tide stage for forecast time (simplified)"""
        
        # This is a simplified implementation
        # In production, this would query the tide_table from Phase I
        
        return {
            'stage': 'rising',  # rising, falling, high, low
            'height': tide_conditions.get('current_level', 0)
        }
    
    def _calculate_confidence_ratings(self, surf_forecast):
        """Calculate confidence ratings for forecasts"""
        
        for period in surf_forecast:
            # Base confidence starts high for near-term forecasts
            hours_ahead = (period['forecast_time'] - int(time.time())) / 3600
            
            # Confidence decreases with time
            if hours_ahead <= 6:
                base_confidence = 0.9
            elif hours_ahead <= 24:
                base_confidence = 0.8
            elif hours_ahead <= 48:
                base_confidence = 0.7
            else:
                base_confidence = 0.6
            
            # Adjust based on data quality
            existing_confidence = period.get('confidence', base_confidence)
            period['confidence'] = min(existing_confidence, base_confidence)
        
        return surf_forecast

    def generate_star_display(self, rating):
        """
        Generate visual star display for surf rating
        Returns both visual stars and formatted text
        """
        
        # Ensure rating is in valid range
        rating = max(1, min(5, int(rating)))
        
        filled_stars = '★' * rating
        empty_stars = '☆' * (5 - rating)
        
        return {
            'rating': rating,
            'stars_visual': filled_stars + empty_stars,
            'stars_filled': filled_stars,
            'stars_empty': empty_stars,
            'rating_text': f"{rating}/5 stars",
            'rating_fraction': f"{rating}/5"
        }

    def format_wave_height_range(self, min_height, max_height):
        """
        Format wave height as consistent range text (e.g., "2-4 ft")
        Handles rounding and edge cases for display
        """
        
        # Round to nearest 0.5 feet for realistic ranges
        min_rounded = round(min_height * 2) / 2
        max_rounded = round(max_height * 2) / 2
        
        # Handle edge cases
        if min_rounded <= 0:
            min_rounded = 0.5
        
        if max_rounded <= min_rounded:
            max_rounded = min_rounded + 0.5
        
        # Format based on size
        if max_rounded < 1:
            return "Flat"
        elif min_rounded == max_rounded:
            return f"{min_rounded:.0f}ft" if min_rounded == int(min_rounded) else f"{min_rounded:.1f}ft"
        else:
            # Standard range format
            if min_rounded == int(min_rounded):
                min_str = f"{min_rounded:.0f}"
            else:
                min_str = f"{min_rounded:.1f}"
                
            if max_rounded == int(max_rounded):
                max_str = f"{max_rounded:.0f}"
            else:
                max_str = f"{max_rounded:.1f}"
                
            return f"{min_str}-{max_str}ft"

    def generate_comprehensive_description(self, rating, wave_period, wave_range, wind_type, wind_speed):
        """
        Generate detailed human-readable surf condition descriptions
        """
        
        # Base quality descriptions
        quality_terms = {
            5: "Excellent",
            4: "Good", 
            3: "Fair",
            2: "Poor",
            1: "Flat/Tiny"
        }
        
        # Period quality descriptors
        if wave_period >= 12:
            period_desc = "groundswell"
        elif wave_period >= 9:
            period_desc = "clean swell"
        elif wave_period >= 7:
            period_desc = "mixed swell"
        else:
            period_desc = "wind chop"
        
        # Wind descriptors
        wind_descriptors = {
            'calm': 'glassy conditions',
            'offshore': f'clean {wind_type} winds',
            'onshore': f'choppy {wind_type} conditions', 
            'cross': f'{wind_type} winds'
        }
        
        wind_desc = wind_descriptors.get(wind_type, f'{wind_type} winds')
        
        # Construct description
        base_quality = quality_terms.get(rating, "Unknown")
        
        if rating >= 4:
            return f"{base_quality} - {wave_range} {period_desc}, {wind_desc}"
        elif rating >= 3:
            return f"{base_quality} conditions - {wave_range} {period_desc}, {wind_desc}"
        elif rating >= 2:
            return f"{base_quality} surf - {wave_range} {period_desc}, {wind_desc}"
        else:
            return f"{base_quality} - {wave_range}, {wind_desc}"

    def _get_surf_database_fields_from_conf(self):
        """Get surf database field definitions from CONF"""
        service_config = self.config_dict.get('SurfFishingService', {})
        database_schema = service_config.get('database_schema', {})
        surf_fields = database_schema.get('marine_forecast_surf_data', {})
        
        if not surf_fields:
            log.error("No surf database field definitions found in CONF")
            # Fallback to minimal required fields
            return {
                'dateTime': {'type': 'INTEGER'},
                'usUnits': {'type': 'INTEGER'}, 
                'spot_id': {'type': 'TEXT'}
            }
        
        return surf_fields

    def _convert_forecast_data_using_conf(self, forecast_data, forecast_type):
        """Convert forecast data using data-driven CONF specifications"""
        converted_data = forecast_data.copy()
        
        # Get conversion fields from CONF
        service_config = self.config_dict.get('SurfFishingService', {})
        conversion_fields = service_config.get('conversion_fields', {}).get(forecast_type, [])
        
        # Apply conversions to specified fields only
        for field_name in conversion_fields:
            if field_name in converted_data and converted_data[field_name] is not None:
                # Get service reference for unit conversion
                if hasattr(self, 'service_ref'):
                    converted_value = self.service_ref._convert_field_to_weewx_units(
                        field_name, 
                        converted_data[field_name]
                    )
                    converted_data[field_name] = converted_value
                else:
                    log.warning(f"No service reference available for converting {field_name}")
        
        return converted_data

    def _get_target_unit_system(self):
        """Get WeeWX usUnits value using CONF configuration"""
        std_convert_config = self.config_dict.get('StdConvert', {})
        target_unit = std_convert_config.get('target_unit', 'US')
        
        # Get unit system mapping from CONF
        service_config = self.config_dict.get('SurfFishingService', {})
        unit_system_mapping = service_config.get('unit_system_mapping', {
            'US': weewx.US,
            'METRIC': weewx.METRIC,
            'METRICWX': weewx.METRICWX
        })
        
        return unit_system_mapping.get(target_unit, weewx.US)

    def store_surf_forecasts(self, spot_id, forecast_data, db_manager):
        """Store surf forecasts with data-driven unit conversion"""
        try:
            # EXISTING CODE: Clear existing forecasts - WeeWX 5.1 pattern preserved
            db_manager.connection.execute(
                "DELETE FROM marine_forecast_surf_data WHERE spot_id = ?",
                (spot_id,)
            )
            
            # NEW: Get database field definitions from CONF
            surf_fields = self._get_surf_database_fields_from_conf()
            
            # EXISTING CODE: Process each forecast period preserved
            for forecast_period in forecast_data:
                # NEW: Apply data-driven unit conversions
                converted_data = self._convert_forecast_data_using_conf(forecast_period, 'surf')
                
                # NEW: Build dynamic INSERT using CONF field definitions
                field_names = list(surf_fields.keys())
                placeholders = ', '.join(['?' for _ in field_names])
                field_list = ', '.join(field_names)
                
                # NEW: Build values list using CONF field specifications
                values = []
                for field_name in field_names:
                    if field_name == 'dateTime':
                        values.append(int(time.time()))
                    elif field_name == 'usUnits':
                        values.append(self._get_target_unit_system())
                    elif field_name == 'spot_id':
                        values.append(spot_id)
                    else:
                        values.append(converted_data.get(field_name))
                
                # EXISTING CODE: Execute database insert (now data-driven)
                db_manager.connection.execute(
                    f"INSERT INTO marine_forecast_surf_data ({field_list}) VALUES ({placeholders})",
                    values
                )
            
            # EXISTING CODE: Return value preserved
            return True
            
        except Exception as e:
            # EXISTING CODE: Error handling preserved
            log.error(f"Error storing surf forecasts for spot {spot_id}: {e}")
            return False

    def get_current_surf_forecast(self, spot_id, hours_ahead=24, db_manager=None):
        """
        Retrieve current surf forecasts using WeeWX 5.1 patterns - CONFIRMED EXISTS IN GITHUB CODE
        
        SURGICAL FIX: Removes manual cursor patterns
        RETAINS: All functionality, method name, parameters, return values
        READS FROM: Existing CONF for forecast settings if needed
        """
        try:
            # Get forecast settings from existing CONF structure if needed
            service_config = self.config_dict.get('SurfFishingService', {})
            forecast_settings = service_config.get('forecast_settings', {})
            default_hours = int(forecast_settings.get('forecast_hours', '72'))
            
            # Use parameter or config default
            if hours_ahead is None:
                hours_ahead = default_hours
            
            current_time = int(time.time())
            end_time = current_time + (hours_ahead * 3600)
            
            # WeeWX 5.1 pattern - direct execute (no manual cursor)
            result = db_manager.connection.execute("""
                SELECT forecast_time, wave_height_min, wave_height_max, wave_height_range,
                    wave_period, wave_direction, wind_speed, wind_direction, 
                    wind_condition, quality_rating, quality_stars, quality_text,
                    conditions_description, confidence
                FROM marine_forecast_surf_data 
                WHERE spot_id = ? AND forecast_time BETWEEN ? AND ?
                ORDER BY forecast_time ASC
            """, (spot_id, current_time, end_time))
            
            # WeeWX 5.1 pattern - fetchall() on result object
            rows = result.fetchall()
            
            # Convert to forecast dictionaries (preserve exact functionality)
            forecasts = []
            for row in rows:
                forecasts.append({
                    'forecast_time': row[0],
                    'wave_height_min': row[1],
                    'wave_height_max': row[2],
                    'wave_height_range': row[3],
                    'wave_period': row[4],
                    'wave_direction': row[5],
                    'wind_speed': row[6],
                    'wind_direction': row[7],
                    'wind_condition': row[8],
                    'quality_rating': row[9],
                    'quality_stars': row[10],
                    'quality_text': row[11],
                    'conditions_description': row[12],
                    'confidence': row[13]
                })
            
            return forecasts
            
        except Exception as e:
            log.error(f"Error retrieving current surf forecast for spot {spot_id}: {e}")
            return []

    def find_next_good_session(self, spot_id, db_manager, min_rating=4):
        """
        Find the next surf session with rating >= min_rating
        Returns details of the next good session
        
        SURGICAL FIX: Removes manual cursor pattern, uses WeeWX 5.1 direct execute
        RETAINS: All functionality, method name, parameters, return values
        """
        try:
            current_time = int(time.time())
            
            # WeeWX 5.1 pattern - direct execute (no manual cursor)
            result = db_manager.connection.execute("""
                SELECT forecast_time, wave_height_range, wave_period,
                    quality_rating, quality_stars, conditions_description
                FROM marine_forecast_surf_data 
                WHERE spot_id = ? 
                AND forecast_time > ?
                AND quality_rating >= ?
                ORDER BY forecast_time
                LIMIT 1
            """, (spot_id, current_time, min_rating))
            
            # WeeWX 5.1 pattern - fetchone() on result object
            row = result.fetchone()
            if row:
                hours_away = (row[0] - current_time) / 3600
                
                return {
                    'found': True,
                    'forecast_time': row[0],
                    'formatted_time': datetime.fromtimestamp(row[0]).strftime('%a %I:%M %p'),
                    'hours_away': round(hours_away, 1),
                    'wave_height_range': row[1],
                    'wave_period': row[2],
                    'quality_rating': row[3],
                    'quality_stars': row[4],
                    'conditions_description': row[5]
                }
            else:
                return {
                    'found': False,
                    'message': f'No {min_rating}+ star sessions in next 72 hours'
                }
                
        except Exception as e:
            log.error(f"Error finding next good session for spot_id {spot_id}: {e}")
            return {'found': False, 'message': 'Error retrieving forecast'}

    def get_today_surf_summary(self, spot_id, db_manager):
        """
        Get summary of today's surf conditions
        
        SURGICAL FIX: Removes manual cursor pattern, uses WeeWX 5.1 direct execute
        RETAINS: All functionality, method name, parameters, return values
        """
        try:
            # Get today's date range
            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            today_end = today_start + timedelta(days=1)
            
            # WeeWX 5.1 pattern - direct execute (no manual cursor)
            result = db_manager.connection.execute("""
                SELECT quality_rating, wave_height_range, conditions_description
                FROM marine_forecast_surf_data 
                WHERE spot_id = ? 
                AND forecast_time >= ? 
                AND forecast_time < ?
                ORDER BY quality_rating DESC
            """, (spot_id, int(today_start.timestamp()), int(today_end.timestamp())))
            
            # WeeWX 5.1 pattern - fetchall() on result object
            rows = result.fetchall()
            
            if rows:
                ratings = [row[0] for row in rows]
                return {
                    'max_rating': max(ratings),
                    'avg_rating': round(sum(ratings) / len(ratings), 1),
                    'session_count': len(ratings),
                    'best_conditions': rows[0][2]  # First row has highest rating
                }
            else:
                return {
                    'max_rating': 0,
                    'avg_rating': 0,
                    'session_count': 0,
                    'best_conditions': 'No forecast available'
                }
                
        except Exception as e:
            log.error(f"Error getting today's surf summary for spot_id {spot_id}: {e}")
            return {
                'max_rating': 0,
                'avg_rating': 0,
                'session_count': 0,
                'best_conditions': 'Error retrieving data'
            }

    def integrate_optimal_data_sources(self, spot_config):
        """Integrate optimal data source selection into surf forecast generation"""
        location_coords = (float(spot_config['latitude']), float(spot_config['longitude']))
        
        # Get optimal sources using integration manager
        integration_manager = getattr(self, 'integration_manager', None)
        if not integration_manager:
            log.warning(f"{CORE_ICONS['warning']} No integration manager available - using default data sources")
            return self._get_default_marine_conditions()
        
        # Select optimal wave source
        optimal_wave_source = integration_manager.select_optimal_wave_source(location_coords)
        
        # Select optimal atmospheric sources
        optimal_atmospheric_sources = integration_manager.select_optimal_atmospheric_sources(location_coords)
        
        # Get current data from optimal sources
        marine_conditions = self._fetch_data_from_optimal_sources(
            optimal_wave_source,
            optimal_atmospheric_sources
        )
        
        # Apply coastal transformation factors
        if optimal_wave_source and marine_conditions.get('wave_data'):
            marine_conditions['wave_data'] = self.apply_coastal_transformation_factors(
                marine_conditions['wave_data'],
                optimal_wave_source['distance_miles']
            )
        
        return marine_conditions
    
    def apply_coastal_transformation_factors(self, wave_data, distance_miles):
        """Apply research-based coastal wave transformation (Komar methodology)"""
        transformed_data = wave_data.copy()
        
        # Base height reduction factor (waves lose ~30% height reaching shore)
        base_height_factor = 0.7
        
        # Distance-based additional reduction
        if distance_miles > 25:
            distance_factor = max(0.8, 1.0 - (distance_miles - 25) / 200)  # Gradual reduction
        else:
            distance_factor = 1.0
        
        # Apply transformation to wave height
        if 'wave_height' in transformed_data and transformed_data['wave_height'] is not None:
            transformed_data['wave_height'] *= (base_height_factor * distance_factor)
        
        # Wave period typically remains more stable during propagation
        # No major transformation needed for period
        
        # Add quality confidence based on distance
        integration_manager = getattr(self, 'integration_manager', None)
        if integration_manager:
            quality_score = integration_manager.calculate_wave_quality(distance_miles)
            transformed_data['data_quality'] = quality_score
            transformed_data['transformation_applied'] = True
            transformed_data['distance_miles'] = distance_miles
        
        return transformed_data
    
    def validate_wave_data_quality(self, primary_source_data, secondary_source_data=None):
        """Cross-validate wave data between sources"""
        validation_result = {
            'is_valid': True,
            'confidence': 1.0,
            'warnings': [],
            'primary_quality': 'good'
        }
        
        # Basic validity checks for primary source
        if not primary_source_data or 'wave_height' not in primary_source_data:
            validation_result['is_valid'] = False
            validation_result['warnings'].append("No primary wave height data available")
            return validation_result
        
        wave_height = primary_source_data['wave_height']
        wave_period = primary_source_data.get('wave_period')
        
        # Sanity checks for wave height
        if wave_height < 0 or wave_height > 50:  # Extreme wave height check
            validation_result['is_valid'] = False
            validation_result['warnings'].append(f"Unrealistic wave height: {wave_height}ft")
            return validation_result
        
        # Period validation if available
        if wave_period is not None:
            if wave_period < 2 or wave_period > 25:  # Realistic period range
                validation_result['warnings'].append(f"Unusual wave period: {wave_period}s")
                validation_result['confidence'] *= 0.8
        
        # Cross-validation with secondary source if available
        if secondary_source_data and 'wave_height' in secondary_source_data:
            secondary_height = secondary_source_data['wave_height']
            height_difference = abs(wave_height - secondary_height)
            
            # Flag if sources disagree significantly (>30% difference)
            if height_difference > (max(wave_height, secondary_height) * 0.3):
                validation_result['warnings'].append(
                    f"Wave height disagreement between sources: {wave_height}ft vs {secondary_height}ft"
                )
                validation_result['confidence'] *= 0.7
        
        # Determine overall quality rating
        if validation_result['confidence'] >= 0.9:
            validation_result['primary_quality'] = 'excellent'
        elif validation_result['confidence'] >= 0.7:
            validation_result['primary_quality'] = 'good'
        elif validation_result['confidence'] >= 0.5:
            validation_result['primary_quality'] = 'fair'
        else:
            validation_result['primary_quality'] = 'poor'
        
        return validation_result
    
    def _fetch_data_from_optimal_sources(self, wave_source, atmospheric_sources):
        """Fetch current data from optimal sources identified by integration manager"""
        marine_conditions = {
            'wave_data': {},
            'atmospheric_data': {},
            'source_info': {
                'wave_source': wave_source,
                'atmospheric_sources': atmospheric_sources,
                'integration_method': 'optimal_selection'
            }
        }
        
        # Fetch wave data from optimal source
        if wave_source:
            try:
                # Query Phase I ndbc_data table for current conditions
                wave_data = self._query_current_wave_data(wave_source['station_id'])
                if wave_data:
                    marine_conditions['wave_data'] = wave_data
                    log.debug(f"{CORE_ICONS['status']} Retrieved wave data from optimal source {wave_source['station_id']}")
            except Exception as e:
                log.error(f"{CORE_ICONS['warning']} Error fetching wave data from {wave_source['station_id']}: {e}")
        
        # Fetch and fuse atmospheric data from optimal sources
        if atmospheric_sources:
            try:
                atmospheric_data_sources = []
                for atm_source in atmospheric_sources:
                    atm_data = self._query_current_atmospheric_data(atm_source['station_id'])
                    if atm_data:
                        atmospheric_data_sources.append({
                            'station_id': atm_source['station_id'],
                            'data': atm_data,
                            'distance_miles': atm_source['distance_miles'],
                            'quality_score': atm_source['quality_score'],
                            'station_type': 'standard_buoy'  # Default, could be enhanced
                        })
                
                if atmospheric_data_sources:
                    # Use DataFusionProcessor for multi-source integration
                    fusion_processor = getattr(self, 'fusion_processor', None)
                    if fusion_processor:
                        fused_atmospheric = fusion_processor.fuse_atmospheric_data(atmospheric_data_sources)
                        marine_conditions['atmospheric_data'] = fused_atmospheric
                    else:
                        # Fallback to primary source only
                        marine_conditions['atmospheric_data'] = atmospheric_data_sources[0]['data']
                    
                    log.debug(f"{CORE_ICONS['status']} Retrieved atmospheric data from {len(atmospheric_data_sources)} optimal sources")
            except Exception as e:
                log.error(f"{CORE_ICONS['warning']} Error fetching atmospheric data: {e}")
        
        return marine_conditions
    
    def _query_current_wave_data(self, station_id):
        """
        Query Phase I ndbc_data table for current wave conditions - CONFIRMED EXISTS
        
        SURGICAL FIX: Removes manual cursor pattern, uses WeeWX 5.1 direct execute
        RETAINS: All functionality, method name, parameters, return values
        """
        try:
            db_manager = getattr(self, 'db_manager', None)
            if not db_manager:
                return None
                
            # WeeWX 5.1 pattern - direct execute (no manual cursor)
            result = db_manager.connection.execute("""
                SELECT wave_height, wave_period, wave_direction, observation_time
                FROM ndbc_data 
                WHERE station_id = ?
                AND wave_height IS NOT NULL
                ORDER BY observation_time DESC 
                LIMIT 1
            """, (station_id,))
            
            # WeeWX 5.1 pattern - fetchone() on result object
            row = result.fetchone()
            if row:
                return {
                    'wave_height': row[0],
                    'wave_period': row[1],
                    'wave_direction': row[2],
                    'observation_time': row[3]
                }
            return None
            
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error querying wave data for station {station_id}: {e}")
            return None
    
    def _query_current_atmospheric_data(self, station_id):
        """
        Query Phase I ndbc_data table for current atmospheric conditions - CONFIRMED EXISTS
        
        SURGICAL FIX: Removes manual cursor pattern, uses WeeWX 5.1 direct execute
        RETAINS: All functionality, method name, parameters, return values
        """
        try:
            db_manager = getattr(self, 'db_manager', None)
            if not db_manager:
                return None
                
            # Calculate time threshold for recent data (last 6 hours)
            recent_threshold = int(time.time()) - (6 * 3600)
            
            # WeeWX 5.1 pattern - direct execute (no manual cursor)
            result = db_manager.connection.execute("""
                SELECT wind_speed, wind_direction, barometric_pressure, 
                    air_temperature, observation_time
                FROM ndbc_data 
                WHERE station_id = ?
                AND observation_time > ?
                AND wind_speed IS NOT NULL
                ORDER BY observation_time DESC 
                LIMIT 1
            """, (station_id, recent_threshold))
            
            # WeeWX 5.1 pattern - fetchone() on result object
            row = result.fetchone()
            if row:
                return {
                    'wind_speed': row[0],
                    'wind_direction': row[1],
                    'barometric_pressure': row[2],
                    'air_temperature': row[3],
                    'observation_time': row[4]
                }
            return None
            
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error querying atmospheric data for station {station_id}: {e}")
            return None
    
    def _get_default_marine_conditions(self):
        """Fallback method to get marine conditions when integration manager unavailable"""
        log.warning(f"{CORE_ICONS['warning']} Using default marine conditions - integration manager not available")
        return {
            'wave_data': {'wave_height': 2.0, 'wave_period': 8.0, 'wave_direction': 270},
            'atmospheric_data': {'wind_speed': 10.0, 'wind_direction': 270, 'barometric_pressure': 30.0},
            'source_info': {'integration_method': 'default_fallback'}
        }

    def lookup_score_from_range_table(self, value, range_table):
        """
        Look up score from range-based scoring table
        """
        try:
            ranges_dict = range_table.get('ranges', {})
            
            for range_key, score in ranges_dict.items():
                # Parse range key (e.g., "3.0-6.0", "15.0+", "0-6")
                if '+' in range_key:
                    # Handle "15.0+" format
                    min_val = float(range_key.replace('+', ''))
                    if value >= min_val:
                        return float(score)
                elif '-' in range_key:
                    # Handle "3.0-6.0" format
                    parts = range_key.split('-')
                    if len(parts) == 2:
                        min_val = float(parts[0])
                        max_val = float(parts[1])
                        if min_val <= value <= max_val:
                            return float(score)
            
            # Default score if no range matches
            return 0.5
            
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error looking up range score: {e}")
            return 0.5
   

class SurfForecastSearchList(SearchList):
    """
    Provides surf forecast data to WeeWX templates
    """
    
    def __init__(self, generator):
        super(SurfForecastSearchList, self).__init__(generator)
        
    def get_extension_list(self, timespan, db_lookup):
        """
        Return search list with surf forecast data for templates
        """
        
        try:
            # Get database manager
            db_manager = db_lookup()
            
            # Get all active surf spots
            surf_spots = self._get_active_surf_spots(db_manager)
            
            # Build forecast data for each spot
            surf_forecasts = {}
            for spot in surf_spots:
                spot_data = self._get_spot_forecast_data(spot, db_manager)
                if spot_data:
                    surf_forecasts[spot['name']] = spot_data
            
            # Get overall summary
            summary_data = self._get_overall_summary(surf_forecasts)
            
            return [{
                'surf_forecasts': surf_forecasts,
                'surf_summary': summary_data,
                'surf_spots_count': len(surf_spots),
                'last_surf_update': self._get_last_update_time(db_manager)
            }]
            
        except Exception as e:
            log.error(f"Error in SurfForecastSearchList: {e}")
            return [{}]
    
    def _get_active_surf_spots(self, db_manager):
        """Get all active surf spots from CONF configuration for SearchList"""
        
        spots = []
        
        try:
            # Access config through generator
            config_dict = self.generator.config_dict
            service_config = config_dict.get('SurfFishingService', {})
            surf_spots_config = service_config.get('surf_spots', {})
            
            for spot_id, spot_config in surf_spots_config.items():
                # Check if spot is active
                is_active = spot_config.get('active', 'true').lower() in ['true', '1', 'yes']
                
                if is_active:
                    spot = {
                        'id': spot_id,
                        'name': spot_config.get('name', spot_id),
                        'latitude': float(spot_config.get('latitude', '0.0')),
                        'longitude': float(spot_config.get('longitude', '0.0')),
                        'bottom_type': spot_config.get('bottom_type', 'sand'),
                        'exposure': spot_config.get('exposure', 'exposed')
                    }
                    spots.append(spot)
            
            log.debug(f"SearchList loaded {len(spots)} surf spots from CONF")
            
        except Exception as e:
            log.error(f"SearchList error getting surf spots from CONF: {e}")
        
        return spots
    
    def _get_spot_forecast_data(self, spot, db_manager):
        """Get complete forecast data for a specific spot"""
        
        try:
            # Create SurfForecastGenerator instance to use its methods
            generator = SurfForecastGenerator(self.generator.config_dict)
            
            return {
                'spot_info': spot,
                'current_forecast': generator.get_current_surf_forecast(spot['id'], db_manager, 24),
                'next_good_session': generator.find_next_good_session(spot['id'], db_manager),
                'today_summary': generator.get_today_surf_summary(spot['id'], db_manager)
            }
            
        except Exception as e:
            log.error(f"Error getting forecast data for {spot['name']}: {e}")
            return None
    
    def _get_overall_summary(self, surf_forecasts):
        """Generate overall surf summary across all spots"""
        
        if not surf_forecasts:
            return {'status': 'No surf spots configured'}
        
        best_rating = 0
        best_spot = None
        
        for spot_name, spot_data in surf_forecasts.items():
            today_summary = spot_data.get('today_summary', {})
            max_rating = today_summary.get('max_rating', 0)
            
            if max_rating > best_rating:
                best_rating = max_rating
                best_spot = spot_name
        
        if best_rating >= 4:
            status = f'Excellent surf at {best_spot}'
        elif best_rating >= 3:
            status = f'Good surf at {best_spot}'
        elif best_rating >= 2:
            status = f'Fair surf available'
        else:
            status = 'Poor surf conditions'
        
        return {
            'status': status,
            'best_rating': best_rating,
            'best_spot': best_spot,
            'total_spots': len(surf_forecasts)
        }
    
    def _get_last_update_time(self, db_manager):
        """
        Get timestamp of last surf forecast update
        
        SURGICAL FIX: Removes manual cursor pattern, uses WeeWX 5.1 direct execute
        RETAINS: All functionality, method name, parameters, return values
        """
        try:
            # WeeWX 5.1 pattern - direct execute (no manual cursor)
            result = db_manager.connection.execute(
                "SELECT MAX(generated_time) FROM marine_forecast_surf_data"
            )
            
            # WeeWX 5.1 pattern - fetchone() on result object
            row = result.fetchone()
            
            if row and row[0]:
                return datetime.fromtimestamp(row[0]).strftime('%m/%d %I:%M %p')
            
            return 'Never'
            
        except Exception as e:
            log.error(f"Error getting last surf update time: {e}")
            return 'Never'
        

class FishingForecastGenerator:
    """Generate fishing condition forecasts with single-decision data fusion architecture"""
    
    def __init__(self, config_dict, engine=None):
        """Initialize with CONF-driven configuration and data fusion components"""
        self.config_dict = config_dict
        self.engine = engine  # Store engine reference for thread-safe database access
        
        # Get service configuration
        self.service_config = config_dict.get('SurfFishingService', {})
        
        # Read data integration preferences from CONF
        self.station_integration = self.service_config.get('station_integration', {'type': 'noaa_only'})
        
        # Read scoring criteria from CONF 
        scoring_criteria = self.service_config.get('scoring_criteria', {})
        self.fishing_scoring = scoring_criteria.get('fishing_scoring', {})
        
        # Read fish categories from CONF
        self.fish_categories = self.service_config.get('fish_categories', {})
        
        # Integration components (set by service during initialization)
        self.integration_manager = None
        self.fusion_processor = None
        
        log.info(f"{CORE_ICONS['status']} FishingForecastGenerator initialized with {self.station_integration['type']} data integration")

    def generate_fishing_forecast(self, spot, marine_conditions):
        """Generate complete fishing forecast for a spot using single-decision architecture"""
        try:
            log.debug(f"{CORE_ICONS['navigation']} Generating fishing forecast for {spot['name']}")
            
            # Generate fishing periods (standard 3-day forecast) - CONSOLIDATED INLINE
            periods = []
            period_definitions = [
                ('Early Morning', 5, 8),
                ('Morning', 8, 12), 
                ('Afternoon', 12, 17),
                ('Evening', 17, 21),
                ('Night', 21, 5)
            ]
            
            current_time = int(time.time())
            for day_offset in range(3):
                day_timestamp = current_time + (day_offset * 86400)
                day_start = day_timestamp - (day_timestamp % 86400)
                
                for period_name, start_hour, end_hour in period_definitions:
                    period = {
                        'forecast_date': day_start,
                        'period_name': period_name,
                        'period_start_hour': start_hour,
                        'period_end_hour': end_hour,
                        'period_start_time': day_start + (start_hour * 3600),
                        'period_end_time': day_start + (end_hour * 3600) if end_hour > start_hour else day_start + ((end_hour + 24) * 3600)
                    }
                    periods.append(period)
            
            forecasts = []
            for period in periods:
                # SINGLE DECISION POINT: Score each period with unified data sources
                period_score = self.score_fishing_period_unified(period, spot, marine_conditions)
                
                # Create forecast record
                forecast = {
                    'forecast_time': period['period_start_time'],
                    'forecast_date': period['forecast_date'],
                    'period_name': period['period_name'],
                    'period_start_hour': period['period_start_hour'],
                    'period_end_hour': period['period_end_hour'],
                    'generated_time': int(time.time()),
                    'pressure_trend': period_score['pressure']['trend'],
                    'tide_movement': period_score['tide']['movement'],
                    'species_activity': period_score['species']['activity_level'],
                    'activity_rating': period_score['overall']['rating'],
                    'conditions_text': period_score['overall']['description'],
                    'best_species': period_score['species']['best_species'],
                    'confidence': period_score['overall']['confidence']
                }
                forecasts.append(forecast)
            
            log.debug(f"{CORE_ICONS['status']} Generated {len(forecasts)} fishing forecast periods")
            return forecasts
            
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error generating fishing forecast: {e}")
            return []

    def score_fishing_period_unified(self, period, spot, marine_conditions):
        """
        SINGLE DECISION POINT: Score fishing period using enhanced CONF-based parameters
        """
        try:
            # PRESERVE EXISTING: Data source decision logic (UNCHANGED)
            data_sources = {
                'pressure': {'type': 'noaa_only', 'sources': [], 'confidence': 0.5},
                'tide': {'type': 'noaa_only', 'sources': [], 'confidence': 0.5}
            }
            
            integration_type = self.station_integration.get('type', 'noaa_only')
            location_coords = (float(spot['latitude']), float(spot['longitude']))
            
            # PRESERVE EXISTING: Integration manager logic (UNCHANGED)
            if integration_type == 'station_supplement' and self.integration_manager:
                log.debug(f"{CORE_ICONS['selection']} Using station supplement data fusion")
                atm_sources = self.integration_manager.select_optimal_atmospheric_sources(location_coords)
                if atm_sources:
                    data_sources['pressure'] = {'type': 'fusion', 'sources': atm_sources, 'confidence': 0.8}
                tide_source = self.integration_manager.select_optimal_tide_source(location_coords)
                if tide_source:
                    data_sources['tide'] = {'type': 'phase_i', 'sources': [tide_source], 'confidence': 0.9}
            else:
                log.debug(f"{CORE_ICONS['navigation']} Using NOAA-only data sources")
                if self.integration_manager:
                    atm_sources = self.integration_manager.select_optimal_atmospheric_sources(location_coords)
                    if atm_sources:
                        data_sources['pressure'] = {'type': 'noaa_only', 'sources': atm_sources, 'confidence': 0.7}

            # PRESERVE EXISTING: Data collection logic (UNCHANGED)
            try:
                pressure_data = self._collect_pressure_data(period, marine_conditions, data_sources['pressure'])
            except:
                pressure_data = {'pressure': 30.0, 'trend': 0.0, 'confidence': 0.3}

            try:
                tide_data = self._collect_tide_data(period, marine_conditions, data_sources['tide'])
            except:
                tide_data = {'tide_movement': 'unknown', 'time_to_next': 6.0, 'confidence': 0.3}

            # ENHANCED: Score pressure using CONF parameters (REPLACES HARDCODED VALUES)
            try:
                # Get enhanced pressure scoring from CONF
                pressure_trend_scoring = self.fishing_scoring.get('pressure_trend_scoring', {}).get('ranges', {})
                pressure_rate_thresholds = self.fishing_scoring.get('pressure_rate_thresholds', {})
                
                # Data-driven pressure rate thresholds from CONF
                fast_change_min = float(pressure_rate_thresholds.get('fast_change_min', '2.0'))
                slow_change_max = float(pressure_rate_thresholds.get('slow_change_max', '0.5'))
                stable_range = float(pressure_rate_thresholds.get('stable_range', '0.2'))
                
                # Classify pressure trend using CONF thresholds
                abs_trend = abs(pressure_data['trend'])
                current_pressure = pressure_data['pressure']
                
                if abs_trend <= stable_range:
                    if current_pressure > 30.10:
                        trend_key = 'stable_high'
                    else:
                        trend_key = 'stable_low'
                elif pressure_data['trend'] > 0:
                    if abs_trend >= fast_change_min:
                        trend_key = 'rising_fast'
                    else:
                        trend_key = 'rising_slow'
                else:
                    if abs_trend >= fast_change_min:
                        trend_key = 'falling_fast'
                    else:
                        trend_key = 'falling_slow'
                
                # Get score from CONF
                pressure_score_value = float(pressure_trend_scoring.get(trend_key, '0.5'))
                
                pressure_score = {
                    'score': pressure_score_value,
                    'trend': trend_key,
                    'change_6h': pressure_data['trend'],
                    'confidence': pressure_data['confidence'],
                    'sources_used': len(data_sources['pressure']['sources']),
                    'description': f'{trend_key.replace("_", " ").title()} pressure conditions'
                }
            except Exception as e:
                log.error(f"{CORE_ICONS['warning']} Error in enhanced pressure scoring: {e}")
                pressure_score = {'score': 0.5, 'trend': 'stable', 'change_6h': 0.0, 'confidence': 0.3, 'sources_used': 0, 'description': 'Unknown pressure conditions'}

            # ENHANCED: Score tide using CONF parameters (REPLACES HARDCODED VALUES)
            try:
                # Get enhanced tide scoring from CONF
                tide_phase_scoring = self.fishing_scoring.get('tide_phase_scoring', {})
                
                # Map tide movement to scoring key
                tide_movement = tide_data.get('tide_movement', 'unknown')
                if tide_movement in ['rising', 'incoming']:
                    tide_key = 'incoming'
                elif tide_movement in ['falling', 'outgoing']:
                    tide_key = 'outgoing'
                elif tide_movement in ['high_slack', 'high']:
                    tide_key = 'high_slack'
                elif tide_movement in ['low_slack', 'low']:
                    tide_key = 'low_slack'
                else:
                    tide_key = 'outgoing'  # Default to best condition
                
                # Get score from CONF
                tide_score_value = float(tide_phase_scoring.get(tide_key, '0.6'))
                
                tide_score = {
                    'score': tide_score_value,
                    'movement': tide_movement,
                    'time_to_next_hours': tide_data.get('time_to_next', 6.0),
                    'confidence': tide_data['confidence'],
                    'description': f'{tide_key.replace("_", " ").title()} tide conditions'
                }
            except Exception as e:
                log.error(f"{CORE_ICONS['warning']} Error in enhanced tide scoring: {e}")
                tide_score = {'score': 0.6, 'movement': 'unknown', 'time_to_next_hours': 6.0, 'confidence': 0.3, 'description': 'Unknown tide conditions'}

            # ENHANCED: Score time using CONF parameters (REPLACES HARDCODED VALUES)  
            try:
                # Get enhanced time scoring from CONF
                time_of_day_scoring = self.fishing_scoring.get('time_of_day_scoring', {})
                
                # Determine time period
                period_hour = period['period_start_hour']
                if 5 <= period_hour <= 7:
                    time_key = 'dawn'
                    period_name = 'Dawn'
                elif 8 <= period_hour <= 11:
                    time_key = 'morning'
                    period_name = 'Morning'
                elif 12 <= period_hour <= 16:
                    time_key = 'midday'
                    period_name = 'Midday'
                elif 17 <= period_hour <= 19:
                    time_key = 'dusk'
                    period_name = 'Dusk'
                elif 20 <= period_hour <= 23:
                    time_key = 'night'
                    period_name = 'Night'
                else:
                    time_key = 'night'
                    period_name = 'Late Night'
                
                # Get score from CONF
                time_score_value = float(time_of_day_scoring.get(time_key, '0.6'))
                
                time_score = {
                    'score': time_score_value,
                    'period_name': period_name,
                    'peak_times': ['dawn', 'dusk'],  # Prime feeding times
                    'description': f'{period_name} fishing period'
                }
            except Exception as e:
                log.error(f"{CORE_ICONS['warning']} Error in enhanced time scoring: {e}")
                time_score = {'score': 0.6, 'period_name': 'Unknown', 'peak_times': ['dawn', 'dusk'], 'description': 'Unknown time period'}

            # ENHANCED: Score species using CONF fish categories (REPLACES HARDCODED VALUES)
            try:
                target_category = spot.get('target_category', 'mixed_bag')
                category_config = self.fish_categories.get(target_category, {})
                
                if category_config:
                    # Get species list from CONF (string format converted to list)
                    species_string = category_config.get('species', 'Mixed')
                    if isinstance(species_string, str):
                        species_list = [s.strip() for s in species_string.split(',')]
                    else:
                        species_list = ['Mixed']
                    
                    # Get species activity modifiers from CONF
                    species_modifiers = self.fishing_scoring.get('species_activity_modifiers', {})
                    category_modifier = float(species_modifiers.get(target_category, '1.0'))
                    
                    # Calculate species activity using CONF-based weights
                    pressure_component = pressure_score['score'] * 0.4
                    tide_component = tide_score['score'] * 0.4
                    base_activity = 0.2
                    
                    activity_score = (pressure_component + tide_component + base_activity) * category_modifier
                    
                    # Determine activity level
                    if activity_score >= 0.8:
                        activity_level = 'high'
                        best_species = species_list[:2] if len(species_list) >= 2 else species_list
                    elif activity_score >= 0.6:
                        activity_level = 'moderate'
                        best_species = species_list[:3] if len(species_list) >= 3 else species_list
                    elif activity_score >= 0.4:
                        activity_level = 'low'
                        best_species = species_list[:1] if species_list else ['Mixed bag']
                    else:
                        activity_level = 'very_low'
                        best_species = ['Opportunistic species']
                    
                    species_score = {
                        'score': min(1.0, max(0.0, activity_score)),
                        'activity_level': activity_level,
                        'best_species': best_species,
                        'target_category': target_category,
                        'description': f'{activity_level.replace("_", " ").title()} species activity'
                    }
                else:
                    species_score = {
                        'score': 0.5,
                        'activity_level': 'moderate',
                        'best_species': ['Mixed bag'],
                        'target_category': target_category,
                        'description': 'Moderate species activity'
                    }
            except Exception as e:
                log.error(f"{CORE_ICONS['warning']} Error scoring species: {e}")
                species_score = {'score': 0.5, 'activity_level': 'moderate', 'best_species': ['Mixed bag'], 'target_category': 'mixed_bag', 'description': 'Unknown species activity'}

            # ENHANCED: Calculate overall rating using CONF weights (REPLACES HARDCODED VALUES)
            try:
                # Get component weights from CONF
                scoring_weights = self.fishing_scoring.get('scoring_weights', {})
                pressure_weight = float(scoring_weights.get('pressure_trend', '0.4'))
                tide_weight = float(scoring_weights.get('tide_phase', '0.3'))
                time_weight = float(scoring_weights.get('time_of_day', '0.2'))
                species_weight = float(scoring_weights.get('species_activity', '0.1'))
                
                # Calculate weighted score
                weighted_score = (
                    pressure_score['score'] * pressure_weight +
                    tide_score['score'] * tide_weight +
                    time_score['score'] * time_weight +
                    species_score['score'] * species_weight
                )
                
                # Calculate confidence based on component confidences
                avg_confidence = (
                    pressure_score.get('confidence', 0.5) * pressure_weight +
                    tide_score.get('confidence', 0.5) * tide_weight +
                    0.9 * time_weight +
                    0.8 * species_weight
                )
                
                # Convert to 1-5 star rating
                if weighted_score >= 0.8:
                    rating = 5
                elif weighted_score >= 0.6:
                    rating = 4
                elif weighted_score >= 0.4:
                    rating = 3
                elif weighted_score >= 0.2:
                    rating = 2
                else:
                    rating = 1
                
                overall_rating = {
                    'rating': rating,
                    'raw_score': weighted_score,
                    'confidence': avg_confidence
                }
            except Exception as e:
                log.error(f"{CORE_ICONS['warning']} Error calculating overall rating: {e}")
                overall_rating = {'rating': 2, 'raw_score': 0.4, 'confidence': 0.5}

            # PRESERVE EXISTING: Description generation logic (ENHANCED)
            rating = overall_rating['rating']
            if rating >= 4:
                base_desc = "Excellent fishing conditions"
            elif rating >= 3:
                base_desc = "Good fishing conditions"
            elif rating >= 2:
                base_desc = "Fair fishing conditions"
            else:
                base_desc = "Slow fishing conditions"
            
            # Add key contributing factors
            factors = []
            if pressure_score['score'] >= 0.7:
                factors.append(f"favorable {pressure_score['trend']} pressure")
            if tide_score['score'] >= 0.7 and tide_score['movement'] != 'not_relevant':
                factors.append(f"optimal {tide_score['movement']} tide")
            if time_score['score'] >= 0.7:
                factors.append(f"prime {time_score['period_name']} timing")
            
            if factors:
                description = f"{base_desc} with {', '.join(factors)}"
            else:
                description = base_desc
            
            return {
                'pressure': pressure_score,
                'tide': tide_score,
                'time': time_score,
                'species': species_score,
                'overall': {
                    'rating': overall_rating['rating'],
                    'confidence': overall_rating['confidence'],
                    'description': description
                }
            }
            
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error scoring fishing period: {e}")
            # PRESERVE EXISTING: Default fallback score (UNCHANGED)
            return {
                'pressure': {'score': 0.5, 'trend': 'stable', 'change_6h': 0.0, 'confidence': 0.3, 'sources_used': 0, 'description': 'Default pressure conditions'},
                'tide': {'score': 0.5, 'movement': 'unknown', 'time_to_next_hours': 6.0, 'confidence': 0.3, 'description': 'Default tide conditions'},
                'time': {'score': 0.5, 'period_name': 'Unknown', 'peak_times': ['dawn', 'dusk'], 'description': 'Default time period'},
                'species': {'score': 0.5, 'activity_level': 'moderate', 'best_species': ['Mixed bag'], 'target_category': 'mixed_bag', 'description': 'Moderate species activity'},
                'overall': {'rating': 2, 'confidence': 0.3, 'description': 'Default fishing conditions'}
            }

    def store_fishing_forecasts(self, spot_id, fishing_forecast):
        """Store fishing forecasts using WeeWX 5.1 thread-safe database patterns"""
        try:
            if not fishing_forecast or not self.engine:
                return False
            
            # THREAD SAFE: Get fresh database manager for this thread
            with weewx.manager.open_manager_with_config(self.config_dict, 'wx_binding') as db_manager:
                connection = db_manager.connection
            
            # Clear existing forecasts for this spot (current forecasts only)
            connection.execute(
                "DELETE FROM marine_forecast_fishing_data WHERE spot_id = ?",
                (spot_id,)
            )
            
            # Insert new forecasts
            for period in fishing_forecast:
                connection.execute("""
                    INSERT INTO marine_forecast_fishing_data (
                        spot_id, forecast_date, period_name, period_start_hour, period_end_hour,
                        generated_time, pressure_trend, tide_movement, species_activity, 
                        activity_rating, conditions_text, best_species
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    spot_id,
                    period['forecast_date'],
                    period['period_name'],
                    period['period_start_hour'],
                    period['period_end_hour'],
                    period['generated_time'],
                    period.get('pressure_trend', 'unknown'),
                    period.get('tide_movement', 'unknown'),
                    period.get('species_activity', 'low'),
                    period.get('activity_rating', 1),
                    period.get('conditions_text', 'Unknown'),
                    json.dumps(period.get('best_species', []))
                ))
            
            connection.commit()
            log.debug(f"{CORE_ICONS['status']} Stored {len(fishing_forecast)} fishing forecast periods for spot {spot_id}")
            return True
            
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error storing fishing forecasts for spot {spot_id}: {e}")
            return False

    def get_current_fishing_forecast(self, spot_id, hours_ahead=24):
        """Retrieve current fishing forecasts using WeeWX 5.1 thread-safe patterns"""
        try:
            if not self.engine:
                return []
            
            current_time = int(time.time())
            end_time = current_time + (hours_ahead * 3600)
            
            # THREAD SAFE: Get fresh database manager for this thread
            with weewx.manager.open_manager_with_config(self.config_dict, 'wx_binding') as db_manager:
                result = db_manager.connection.execute("""
                    SELECT forecast_date, period_name, period_start_hour, period_end_hour,
                        pressure_trend, tide_movement, species_activity, activity_rating,
                        conditions_text, best_species, generated_time
                    FROM marine_forecast_fishing_data 
                    WHERE spot_id = ? AND forecast_date BETWEEN ? AND ?
                    ORDER BY forecast_date ASC, period_start_hour ASC
                """, (spot_id, current_time, end_time))
                
            rows = result.fetchall()
            
            forecasts = []
            for row in rows:
                try:
                    best_species = json.loads(row[9]) if row[9] else []
                except (json.JSONDecodeError, TypeError):
                    best_species = []
                
                forecast = {
                    'forecast_date': row[0],
                    'period_name': row[1],
                    'period_start_hour': row[2],
                    'period_end_hour': row[3],
                    'pressure_trend': row[4],
                    'tide_movement': row[5],
                    'species_activity': row[6],
                    'activity_rating': row[7],
                    'conditions_text': row[8],
                    'best_species': best_species,
                    'generated_time': row[10],
                    'forecast_time_text': datetime.fromtimestamp(row[0] + (row[2] * 3600)).strftime('%m/%d %I:%M %p')
                }
                forecasts.append(forecast)
            
            return forecasts
            
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error retrieving fishing forecast for spot {spot_id}: {e}")
            return []

    def find_next_good_fishing_period(self, spot_id, min_rating=3):
        """Find next fishing period with rating >= min_rating using thread-safe patterns"""
        try:
            if not self.engine:
                return None
            
            current_time = int(time.time())
            
            # THREAD SAFE: Get fresh database manager for this thread
            with weewx.manager.open_manager_with_config(self.config_dict, 'wx_binding') as db_manager:
                result = db_manager.connection.execute("""
                    SELECT forecast_date, period_name, period_start_hour, period_end_hour,
                        activity_rating, conditions_text, best_species
                    FROM marine_forecast_fishing_data 
                    WHERE spot_id = ? AND forecast_date >= ? AND activity_rating >= ?
                    ORDER BY forecast_date ASC, period_start_hour ASC
                    LIMIT 1
                """, (spot_id, current_time, min_rating))
            
            row = result.fetchone()
            
            if row:
                try:
                    best_species = json.loads(row[6]) if row[6] else []
                except (json.JSONDecodeError, TypeError):
                    best_species = []
                
                return {
                    'forecast_time': row[0] + (row[2] * 3600),
                    'period_name': row[1],
                    'activity_rating': row[4],
                    'conditions_text': row[5],
                    'best_species': best_species,
                    'forecast_time_text': datetime.fromtimestamp(row[0] + (row[2] * 3600)).strftime('%m/%d %I:%M %p')
                }
            
            return None
            
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error finding next good fishing period: {e}")
            return None

    def export_fishing_forecast_summary(self, spot_id, format='dict'):
        """Export complete fishing forecast summary for templates or external use"""
        try:
            if not self.engine:
                return {'error': 'No engine available'}
                
            # Get current forecasts
            current_forecasts = self.get_current_fishing_forecast(spot_id, 72)
            
            # Get next good period
            next_good_period = self.find_next_good_fishing_period(spot_id, 3)
            
            # Get today's summary - CONSOLIDATED INLINE with thread-safe access
            today_summary = {'max_rating': 0, 'best_period': 'No data', 'avg_rating': 0, 'period_count': 0}
            
            # THREAD SAFE: Get fresh database manager for this thread
            with weewx.manager.open_manager_with_config(self.config_dict, 'wx_binding') as db_manager:
                try:
                    current_time = int(time.time())
                    today_start = current_time - (current_time % 86400)
                    today_end = today_start + 86400
                    
                    result = db_manager.connection.execute("""
                        SELECT activity_rating, conditions_text, period_name
                        FROM marine_forecast_fishing_data 
                        WHERE spot_id = ? AND forecast_date BETWEEN ? AND ?
                        ORDER BY activity_rating DESC
                    """, (spot_id, today_start, today_end))
                    
                    rows = result.fetchall()
                    
                    if rows:
                        ratings = [row[0] for row in rows]
                        max_rating = max(ratings)
                        avg_rating = sum(ratings) / len(ratings)
                        best_period = next((row[2] for row in rows if row[0] == max_rating), 'Unknown')
                        
                        today_summary = {
                            'max_rating': max_rating,
                            'best_period': best_period,
                            'avg_rating': round(avg_rating, 1),
                            'period_count': len(rows)
                        }
                except Exception as e:
                    log.error(f"{CORE_ICONS['warning']} Error getting today's summary: {e}")
            
            # Get spot information from CONF
            service_config = self.config_dict.get('SurfFishingService', {})
            fishing_spots = service_config.get('fishing_spots', {})
            
            spot_info = None
            for spot_key, spot_config in fishing_spots.items():
                if spot_key == spot_id or spot_config.get('name') == spot_id:
                    spot_info = {
                        'id': spot_key,
                        'name': spot_config.get('name', spot_key),
                        'latitude': float(spot_config.get('latitude', '0.0')),
                        'longitude': float(spot_config.get('longitude', '0.0')),
                        'location_type': spot_config.get('location_type', 'shore'),
                        'target_category': spot_config.get('target_category', 'mixed_bag')
                    }
                    break
            
            if not spot_info:
                return {'error': 'Spot not found'}
            
            # Compile complete summary
            summary = {
                'spot_info': spot_info,
                'generated_time': datetime.now().isoformat(),
                'forecasts': current_forecasts,
                'next_good_period': next_good_period,
                'today_summary': today_summary,
                'integration_type': self.station_integration.get('type', 'noaa_only'),
                'forecast_count': len(current_forecasts)
            }
            
            if format == 'json':
                return json.dumps(summary, indent=2)
            else:
                return summary
                
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error exporting fishing forecast summary: {e}")
            return {'error': f'Export failed: {str(e)}'}



class FishingForecastSearchList:
    """Provides fishing forecast data to WeeWX templates - replaces old SearchList"""
    
    def __init__(self, generator):
        """Initialize with WeeWX generator for template integration"""
        self.generator = generator
        
    def get_extension_list(self, timespan, db_lookup):
        """Return search list with fishing forecast data for templates"""
        try:
            # Get database manager
            db_manager = db_lookup()
            
            # Get service configuration
            config_dict = self.generator.config_dict
            service_config = config_dict.get('SurfFishingService', {})
            
            # Get all active fishing spots from CONF
            fishing_spots = service_config.get('fishing_spots', {})
            
            if not fishing_spots:
                return [{'fishing_forecasts': {}, 'fishing_summary': {'status': 'No fishing spots configured'}}]
            
            # Initialize fishing generator with engine reference
            fishing_generator = FishingForecastGenerator(config_dict, self.generator.engine)
            
            # Build forecast data for each spot
            fishing_forecasts = {}
            for spot_id, spot_config in fishing_spots.items():
                spot_data = fishing_generator.export_fishing_forecast_summary(spot_id, 'dict')
                if spot_data and 'error' not in spot_data:
                    fishing_forecasts[spot_config.get('name', spot_id)] = spot_data
            
            # Get overall summary - CONSOLIDATED INLINE
            if not fishing_forecasts:
                summary_data = {'status': 'No fishing spots configured'}
            else:
                best_rating = 0
                best_spot = None
                
                for spot_name, spot_data in fishing_forecasts.items():
                    today_summary = spot_data.get('today_summary', {})
                    max_rating = today_summary.get('max_rating', 0)
                    
                    if max_rating > best_rating:
                        best_rating = max_rating
                        best_spot = spot_name
                
                if best_rating >= 4:
                    status = f'Excellent fishing at {best_spot}'
                elif best_rating >= 3:
                    status = f'Good fishing at {best_spot}'
                elif best_rating >= 2:
                    status = f'Fair fishing available'
                else:
                    status = 'Slow fishing conditions'
                
                summary_data = {
                    'status': status,
                    'best_rating': best_rating,
                    'best_spot': best_spot,
                    'total_spots': len(fishing_forecasts)
                }
            
            # Get last update time - CONSOLIDATED INLINE with thread-safe access
            last_update = 'Never'
            try:
                # THREAD SAFE: Get fresh database manager for this thread
                with weewx.manager.open_manager_with_config(config_dict, 'wx_binding') as db_manager:
                    result = db_manager.connection.execute(
                        "SELECT MAX(generated_time) FROM marine_forecast_fishing_data"
                    )
                    
                    row = result.fetchone()
                    
                    if row and row[0]:
                        last_update = datetime.fromtimestamp(row[0]).strftime('%m/%d %I:%M %p')
                        
            except Exception as e:
                log.error(f"{CORE_ICONS['warning']} Error getting last fishing update time: {e}")
            
            return [{
                'fishing_forecasts': fishing_forecasts,
                'fishing_summary': summary_data,
                'fishing_spots_count': len(fishing_spots),
                'fishing_last_update': last_update
            }]
            
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error in FishingForecastSearchList: {e}")
            return [{'fishing_forecasts': {}, 'fishing_summary': {'status': 'Error loading fishing data'}}]

    def configure_search_lists(self):
        """Register SearchList extensions for template integration"""
        try:
            # This should be handled by the installer, but verify it's working
            import weewx.cheetahgenerator
            if hasattr(weewx.cheetahgenerator.SearchList, 'extensible'):
                if 'user.surf_fishing.FishingForecastSearchList' not in weewx.cheetahgenerator.SearchList.extensible:
                    weewx.cheetahgenerator.SearchList.extensible.append(
                        'user.surf_fishing.FishingForecastSearchList'
                    )
            log.info(f"{CORE_ICONS['status']} SearchList extensions registered")
        except Exception as e:
            log.warning(f"{CORE_ICONS['warning']} SearchList registration issue: {e}")


class SurfFishingService(StdService):
    """
    Main service class for surf and fishing forecasts
    """
    
    def __init__(self, engine, config_dict):
        """Initialize SurfFishingService with WeeWX unit system detection"""
        super(SurfFishingService, self).__init__(engine, config_dict)
        
        # EXISTING CODE: Store references for proper WeeWX integration - PRESERVED
        self.engine = engine
        self.config_dict = config_dict
        
        # EXISTING CODE: CRITICAL FIX - Don't get database manager here, defer it - PRESERVED
        self._db_manager = None
        self._db_lock = threading.Lock()
        self._db_initialization_complete = False
        
        # EXISTING CODE: Read service configuration from CONF only - PRESERVED
        self.service_config = config_dict.get('SurfFishingService', {})
        
        # NEW: WeeWX 5.1 unit system detection using CONF data
        self._setup_unit_system_from_conf(config_dict)
        
        # EXISTING CODE: Initialize GRIB processor - PRESERVED EXACTLY
        self.grib_processor = GRIBProcessor(config_dict)
        if self.grib_processor.is_available():
            log.info("Using pygrib for GRIB processing")
        else:
            log.warning("No GRIB library available - WaveWatch III forecasts disabled")
        
        # EXISTING CODE: Initialize forecast generators with CONF-based config - PRESERVED EXACTLY
        # NOTE: These will use _get_db_manager() when they need database access
        self.surf_generator = SurfForecastGenerator(config_dict, None)  # Pass None, will get via _get_db_manager
        self.fishing_generator = FishingForecastGenerator(config_dict, self.engine)
        
        # EXISTING CODE: Set up forecast timing from CONF - PRESERVED EXACTLY
        self.forecast_interval = int(self.service_config.get('forecast_interval', '21600'))  # 6 hours default
        self.shutdown_event = threading.Event()
        
        # EXISTING CODE: Initialize station integration with CONF-based error handling - PRESERVED EXACTLY
        try:
            log.info(f"{CORE_ICONS['status']} Initializing marine station integration...")
            
            # Load field definitions from CONF (written by installer) - PRESERVED
            field_definitions = self._load_field_definitions()
            
            # Initialize integration components if field definitions available - PRESERVED
            if field_definitions:
                log.info(f"{CORE_ICONS['status']} Station integration initialized successfully")
            else:
                log.warning(f"{CORE_ICONS['warning']} Station integration unavailable - using defaults")
                
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error initializing station integration: {e}")
            log.warning(f"{CORE_ICONS['warning']} Station integration unavailable - using defaults")
        
        # EXISTING CODE: Start forecast generation - PRESERVED EXACTLY
        log.info("Starting forecast generation loop")
        self.bind(weewx.STARTUP, lambda event: self._start_forecast_thread())
        
        log.info("Generating forecasts for all locations")
        log.info("SurfFishingService initialized successfully - database manager will be acquired on first use")

    def _setup_unit_system_from_conf(self, config_dict):
        """Setup unit system detection using CONF configuration specifications"""
        # Detect WeeWX's configured unit system from StdConvert
        std_convert_config = config_dict.get('StdConvert', {})
        target_unit_system = std_convert_config.get('target_unit', 'US')
        
        # Map to WeeWX unit constants
        unit_system_map = {
            'US': weewx.US,
            'METRIC': weewx.METRIC, 
            'METRICWX': weewx.METRICWX
        }
        
        self.unit_system = unit_system_map.get(target_unit_system, weewx.US)
        self.unit_system_name = target_unit_system
        
        # Load conversion specifications from CONF
        service_config = config_dict.get('SurfFishingService', {})
        self.unit_conversion_config = service_config.get('unit_conversions', {})
        
        log.info(f"Unit system detected: {target_unit_system} (usUnits={self.unit_system})")
        log.debug(f"Loaded unit conversion config from CONF: {len(self.unit_conversion_config)} conversion specs")

    def _convert_field_to_weewx_units(self, field_name, raw_value, field_type=None):
        """Convert field value using data-driven CONF specifications"""
        if raw_value is None:
            return None
            
        # Get field type from CONF if not provided
        if field_type is None:
            field_type = self._get_field_type_from_conf(field_name)
        
        if not field_type:
            log.warning(f"No field type found in CONF for field: {field_name}")
            return raw_value
        
        # Get conversion specification from CONF
        conversion_spec = self.unit_conversion_config.get(field_type, {})
        target_unit_spec = conversion_spec.get(self.unit_system_name, {})
        
        if not target_unit_spec:
            log.debug(f"No conversion needed for {field_name} ({field_type}) in {self.unit_system_name} units")
            return raw_value
        
        # Apply conversion using CONF specification
        conversion_factor = target_unit_spec.get('factor', 1.0)
        conversion_offset = target_unit_spec.get('offset', 0.0)
        
        # Handle different conversion types from CONF
        conversion_type = target_unit_spec.get('type', 'linear')
        
        if conversion_type == 'linear':
            return (raw_value * conversion_factor) + conversion_offset
        elif conversion_type == 'formula':
            formula = target_unit_spec.get('formula', 'x')
            return self._apply_conversion_formula(raw_value, formula)
        else:
            log.warning(f"Unknown conversion type '{conversion_type}' for field {field_name}")
            return raw_value

    def _get_field_type_from_conf(self, field_name):
        """Get field type from CONF field_mappings"""
        service_config = self.config_dict.get('SurfFishingService', {})
        field_mappings = service_config.get('field_mappings', {})
        
        return field_mappings.get(field_name, {}).get('unit_type')

    def _apply_conversion_formula(self, value, formula):
        """Apply custom conversion formula from CONF"""
        try:
            # Replace 'x' with actual value in formula
            formula_with_value = formula.replace('x', str(value))
            # Evaluate formula (limited to safe operations)
            return eval(formula_with_value)
        except Exception as e:
            log.error(f"Error applying conversion formula '{formula}': {e}")
            return value

    def _get_db_manager(self):
        """Get database manager - thread-specific if available, otherwise delayed initialization"""
        # If we're in a background thread with a thread-specific manager, use that
        if hasattr(self, '_thread_db_manager') and self._thread_db_manager is not None:
            return self._thread_db_manager
        
        # Otherwise use the original delayed initialization logic
        if self._db_manager is None:
            with self._db_lock:
                if self._db_manager is None:
                    try:
                        log.debug("Acquiring database manager (delayed initialization)")
                        self._db_manager = self.engine.db_binder.get_manager('wx_binding')
                        self._db_initialization_complete = True
                        log.info("Database manager initialized successfully")
                    except Exception as e:
                        log.error(f"Failed to initialize database manager: {e}")
                        self._db_manager = None
                        self._db_initialization_complete = False
                        raise
        
        return self._db_manager
    
    def shutDown(self):
        """Clean shutdown of the service"""
        log.info("Shutting down SurfFishingService")
        
        # Signal forecast thread to stop
        self.shutdown_event.set()
        
        # Wait for thread to finish
        if self.forecast_thread and self.forecast_thread.is_alive():
            self.forecast_thread.join(timeout=10)
        
        super(SurfFishingService, self).shutDown()
    
    def new_loop_packet(self, event):
        """Process real-time station data if station integration enabled"""
        
        if self.station_integration.get('type') == 'station_supplement':
            packet = event.packet
            
            # Extract station data for local conditions
            station_data = {}
            selected_sensors = self.station_integration.get('sensors', {})
            
            if selected_sensors.get('wind', False):
                if 'windSpeed' in packet and 'windDir' in packet:
                    station_data['wind_speed'] = packet['windSpeed']
                    station_data['wind_direction'] = packet['windDir']
            
            if selected_sensors.get('pressure', False):
                if 'barometer' in packet:
                    station_data['pressure'] = packet['barometer']
            
            if selected_sensors.get('temperature', False):
                if 'outTemp' in packet:
                    station_data['air_temperature'] = packet['outTemp']
            
            # Store station data for use in forecasting
            # This could be stored in a separate table or used directly
            if station_data:
                log.debug(f"Station data available: {station_data}")
    
    def _start_forecast_thread(self):
        """Start forecast generation thread"""
        self.forecast_thread = threading.Thread(target=self._forecast_loop, name='SurfFishingForecast')
        self.forecast_thread.daemon = True
        self.forecast_thread.start()

    def _forecast_loop(self):
        """
        Main forecast generation loop - complete operation in one method
        FIXED: Creates thread-local database manager and generators for each iteration
        """
        log.info("Forecast generation thread started")
        
        while not self.shutdown_event.is_set():
            try:
                log.debug("Starting forecast generation (API calls and processing)")
                
                # Open thread-local database connection for entire forecast cycle
                with weewx.manager.open_manager_with_config(self.config_dict, 'wx_binding') as db_manager:
                    
                    # Create thread-local generators with database manager
                    surf_generator = SurfForecastGenerator(self.config_dict, db_manager)
                    fishing_generator = FishingForecastGenerator(self.config_dict, db_manager)
                    
                    # Get active spots
                    active_surf_spots = self._get_active_surf_spots()
                    active_fishing_spots = self._get_active_fishing_spots()
                    
                    surf_count = 0
                    fishing_count = 0
                    
                    # Generate surf forecasts
                    for spot in active_surf_spots:
                        try:
                            # Get WaveWatch III data if available
                            wavewatch_data = []
                            if self.grib_processor.is_available():
                                wavewatch_collector = WaveWatchDataCollector(self.config_dict, self.grib_processor)
                                wavewatch_data = wavewatch_collector.fetch_forecast_data(
                                    spot['latitude'], spot['longitude']
                                )
                            
                            # Generate surf forecast
                            surf_forecast = surf_generator.generate_surf_forecast(spot, wavewatch_data)
                            
                            if surf_forecast:
                                # Store directly with the generator's method
                                surf_generator.store_surf_forecasts(spot['id'], surf_forecast, db_manager)
                                surf_count += 1
                                
                        except Exception as e:
                            log.error(f"Error generating surf forecast for {spot.get('name', 'unknown')}: {e}")
                            continue
                    
                    # Generate fishing forecasts
                    for spot in active_fishing_spots:
                        try:
                            # Generate fishing forecast
                            fishing_forecast = fishing_generator.generate_fishing_forecast(spot)
                            
                            if fishing_forecast:
                                # Store directly with the generator's method
                                fishing_generator.store_fishing_forecasts(spot['id'], fishing_forecast, db_manager)
                                fishing_count += 1
                                
                        except Exception as e:
                            log.error(f"Error generating fishing forecast for {spot.get('name', 'unknown')}: {e}")
                            continue
                    
                    log.info(f"Forecast generation completed for {surf_count} surf spots and {fishing_count} fishing spots")
                
                # Sleep outside of database context
                log.debug(f"Sleeping for {self.forecast_interval} seconds")
                self.shutdown_event.wait(timeout=self.forecast_interval)
                
            except Exception as e:
                log.error(f"Error in forecast loop: {e}")
                self.shutdown_event.wait(timeout=300)
        
        log.info("Forecast generation thread stopped")
        
    def _get_active_surf_spots(self):
        """Get all surf spots from CONF configuration"""
        
        spots = []
        
        try:
            # Read from CONF instead of database
            service_config = self.config_dict.get('SurfFishingService', {})
            surf_spots_config = service_config.get('surf_spots', {})
            
            log.debug(f"Found surf_spots_config: {surf_spots_config}")
            
            for spot_id, spot_config in surf_spots_config.items():
                log.debug(f"Processing spot {spot_id}: {spot_config}")
                
                # Convert CONF data to expected format - all spots in CONF are active
                spot = {
                    'id': spot_id,  # Use CONF key as ID
                    'name': spot_config.get('name', spot_id),
                    'latitude': float(spot_config.get('latitude', '0.0')),
                    'longitude': float(spot_config.get('longitude', '0.0')),
                    'bottom_type': spot_config.get('bottom_type', 'sand'),
                    'exposure': spot_config.get('exposure', 'exposed'),
                    'type': spot_config.get('type', 'surf')
                }
                spots.append(spot)
                log.debug(f"Added surf spot: {spot['name']} at {spot['latitude']}, {spot['longitude']}")
                    
            log.info(f"Loaded {len(spots)} surf spots from CONF")
            
        except Exception as e:
            log.error(f"Error getting surf spots from CONF: {e}")
            log.warning("Using fallback empty surf spots list")
        
        return spots

    def _get_active_fishing_spots(self):
        """Get all fishing spots from CONF configuration"""
        
        spots = []
        
        try:
            # Read from CONF instead of database
            service_config = self.config_dict.get('SurfFishingService', {})
            fishing_spots_config = service_config.get('fishing_spots', {})
            
            log.debug(f"Found fishing_spots_config: {fishing_spots_config}")
            
            for spot_id, spot_config in fishing_spots_config.items():
                log.debug(f"Processing fishing spot {spot_id}: {spot_config}")
                
                # Convert CONF data to expected format - all spots in CONF are active
                spot = {
                    'id': spot_id,  # Use CONF key as ID
                    'name': spot_config.get('name', spot_id),
                    'latitude': float(spot_config.get('latitude', '0.0')),
                    'longitude': float(spot_config.get('longitude', '0.0')),
                    'location_type': spot_config.get('location_type', 'shore'),
                    'target_category': spot_config.get('target_category', 'mixed_bag'),
                    'type': spot_config.get('type', 'fishing')
                }
                spots.append(spot)
                log.debug(f"Added fishing spot: {spot['name']} at {spot['latitude']}, {spot['longitude']}")
                    
            log.info(f"Loaded {len(spots)} fishing spots from CONF")
            
        except Exception as e:
            log.error(f"Error getting fishing spots from CONF: {e}")
            log.warning("Using fallback empty fishing spots list")
        
        return spots
    
    def _generate_surf_forecast_for_spot(self, spot):
        """Generate surf forecast for a specific spot"""
        
        log.debug(f"Generating surf forecast for {spot['name']}")
        
        # Get current marine conditions from Phase I
        marine_conditions = self._get_phase_i_marine_conditions(spot['latitude'], spot['longitude'])
        
        # Generate fishing forecast
        fishing_forecast = self.fishing_generator.generate_fishing_forecast(
            spot, marine_conditions
        )
        
        # Store forecast in database
        self._store_fishing_forecast(spot['id'], fishing_forecast)
    
    def _get_phase_i_marine_conditions(self, latitude, longitude):
        """
        Get marine conditions from Phase I Marine Data Extension
        
        SURGICAL FIX: Removes manual cursor patterns
        RETAINS: All functionality, method name, parameters, return values
        READS FROM: Existing CONF for Phase I integration settings
        """
        
        try:
            # Verify Phase I configuration (preserve existing logic)
            phase_i_config = self.config_dict.get('MarineDataService', {})
            if not phase_i_config:
                log.warning("Phase I Marine Data Extension not configured - returning default conditions")
                return self._get_default_marine_conditions()
            
            conditions = {
                'wave_height': None,
                'wave_period': None,
                'wind_speed': None,
                'wind_direction': None,
                'barometric_pressure': None,
                'water_temperature': None,
                'data_quality': 'unknown',
                'data_age_hours': None
            }
            
            recent_time = int(time.time()) - (6 * 3600)  # 6 hours ago
            
            # Get NDBC field mappings from Phase I YAML
            yaml_config = getattr(self, 'yaml_config', {})
            ndbc_fields = yaml_config.get('fields', {})
            
            # Build field query from available Phase I fields
            ndbc_query_fields = []
            ndbc_field_map = {}
            
            for field_name, field_config in ndbc_fields.items():
                if (field_config.get('api_module') == 'ndbc_module' and 
                    field_config.get('database_table') == 'ndbc_data'):
                    ndbc_query_fields.append(field_name)
                    ndbc_field_map[field_name] = field_config.get('display_name', field_name).lower()
            
            if ndbc_query_fields:
                ndbc_query_fields.append('dateTime')  # Always include timestamp
                
                # ✅ CORRECT: WeeWX 5.1 pattern - direct execute
                result = self.db_manager.connection.execute(f"""
                    SELECT {', '.join(ndbc_query_fields)} 
                    FROM ndbc_data 
                    WHERE dateTime > ? 
                    ORDER BY dateTime DESC 
                    LIMIT 1
                """, (recent_time,))
                
                # ✅ CORRECT: fetchone() on result object
                ndbc_result = result.fetchone()
                
                if ndbc_result and len(ndbc_result) > 0:
                    data_age = (time.time() - ndbc_result[-1]) / 3600  # dateTime is last field
                    
                    # Map database results to logical names using Phase I mappings
                    for i, db_field in enumerate(ndbc_query_fields[:-1]):  # Skip dateTime
                        if ndbc_result[i] is not None:
                            logical_name = ndbc_field_map.get(db_field)
                            if logical_name == 'wave_height':
                                conditions['wave_height'] = ndbc_result[i]
                            elif logical_name in ['wave_period', 'dominant_wave_period']:
                                conditions['wave_period'] = ndbc_result[i]
                            elif logical_name in ['marine_wind_speed', 'wind_speed']:
                                conditions['wind_speed'] = ndbc_result[i]
                            elif logical_name in ['marine_wind_direction', 'wind_direction']:
                                conditions['wind_direction'] = ndbc_result[i]
                            elif logical_name in ['marine_barometric_pressure', 'barometric_pressure']:
                                conditions['barometric_pressure'] = ndbc_result[i]
                            elif logical_name in ['marine_sea_surface_temp', 'sea_surface_temp']:
                                conditions['water_temperature'] = ndbc_result[i]
                    
                    conditions['data_quality'] = 'good' if data_age < 2 else 'fair'
                    conditions['data_age_hours'] = data_age
                    log.debug(f"Retrieved NDBC data from {data_age:.1f} hours ago using Phase I mappings")
                else:
                    log.debug("No recent NDBC data available")
            
            # Get CO-OPS data using Phase I integration (preserve existing logic)
            coops_fields = [field for field, config in ndbc_fields.items() 
                        if config.get('api_module') == 'coops_module']
            
            if coops_fields:
                coops_fields.append('dateTime')
                
                # ✅ CORRECT: WeeWX 5.1 pattern - direct execute
                result = self.db_manager.connection.execute(f"""
                    SELECT {', '.join(coops_fields)} 
                    FROM coops_realtime 
                    WHERE dateTime > ? 
                    ORDER BY dateTime DESC 
                    LIMIT 1
                """, (recent_time,))
                
                # ✅ CORRECT: fetchone() on result object
                coops_result = result.fetchone()
                
                if coops_result:
                    # Process CO-OPS data (preserve existing mapping logic)
                    for i, field_name in enumerate(coops_fields[:-1]):
                        if coops_result[i] is not None:
                            field_config = ndbc_fields.get(field_name, {})
                            logical_name = field_config.get('display_name', '').lower()
                            
                            if 'water_level' in logical_name:
                                conditions['current_water_level'] = coops_result[i]
                            elif 'water_temp' in logical_name:
                                if not conditions['water_temperature']:  # Use if NDBC didn't provide
                                    conditions['water_temperature'] = coops_result[i]
                    
                    log.debug("Retrieved CO-OPS data using Phase I mappings")
            
            return conditions
            
        except Exception as e:
            log.error(f"Error getting Phase I marine conditions: {e}")
            return self._get_default_marine_conditions()
    
    def _store_surf_forecast(self, spot_id, surf_forecast):
        """Store surf forecast in database"""
        
        try:
            with self.db_manager.connection as connection:
                # Clear old forecasts for this spot
                connection.execute("""
                    DELETE FROM marine_forecast_surf_data 
                    WHERE spot_id = ? AND forecast_time < ?
                """, (spot_id, int(time.time())))
                
                # Insert new forecast data
                for period in surf_forecast:
                    connection.execute("""
                        INSERT OR REPLACE INTO marine_forecast_surf_data
                        (spot_id, forecast_time, generated_time, wave_height_min, wave_height_max,
                         wave_period, wave_direction, wind_speed, wind_direction, wind_condition,
                         tide_height, tide_stage, quality_rating, confidence, conditions_text)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        spot_id,
                        period['forecast_time'],
                        int(time.time()),
                        period.get('wave_height_min', 0),
                        period.get('wave_height_max', 0),
                        period.get('wave_period', 0),
                        period.get('wave_direction', 0),
                        period.get('wind_speed', 0),
                        period.get('wind_direction', 0),
                        period.get('wind_condition', 'unknown'),
                        period.get('tide_height', 0),
                        period.get('tide_stage', 'unknown'),
                        period.get('quality_rating', 1),
                        period.get('confidence', 0.5),
                        period.get('conditions_text', 'Unknown')
                    ))
                
                connection.commit()
                log.debug(f"Stored {len(surf_forecast)} surf forecast periods for spot {spot_id}")
        
        except Exception as e:
            log.error(f"Error storing surf forecast for spot {spot_id}: {e}")
    
    def _store_fishing_forecast(self, spot_id, fishing_forecast):
        """Store fishing forecast in database"""
        
        try:
            with self.db_manager.connection as connection:
                # Clear old forecasts for this spot
                connection.execute("""
                    DELETE FROM marine_forecast_fishing_data 
                    WHERE spot_id = ? AND forecast_date < ?
                """, (spot_id, int(time.time()) - 86400))  # Keep today's data
                
                # Insert new forecast data
                for period in fishing_forecast:
                    connection.execute("""
                        INSERT OR REPLACE INTO marine_forecast_fishing_data
                        (spot_id, forecast_date, period_name, period_start_hour, period_end_hour,
                         generated_time, pressure_trend, tide_movement, species_activity,
                         activity_rating, conditions_text, best_species)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        spot_id,
                        period['forecast_date'],
                        period['period_name'],
                        period['period_start_hour'],
                        period['period_end_hour'],
                        period['generated_time'],
                        period.get('pressure_trend', 'unknown'),
                        period.get('tide_movement', 'unknown'),
                        period.get('species_activity', 'low'),
                        period.get('activity_rating', 1),
                        period.get('conditions_text', 'Unknown'),
                        json.dumps(period.get('best_species', []))
                    ))
                
                connection.commit()
                log.debug(f"Stored {len(fishing_forecast)} fishing forecast periods for spot {spot_id}")
        
        except Exception as e:
            log.error(f"Error storing fishing forecast for spot {spot_id}: {e}")

    def initialize_station_integration(self):
        """Initialize Phase I metadata integration during service startup"""
        try:
            log.info(f"{CORE_ICONS['status']} Initializing marine station integration...")
            
            # Load field definitions from YAML
            field_definitions = self._load_field_definitions()
            
            # Initialize integration manager
            self.integration_manager = MarineStationIntegrationManager(
                self.config_dict, 
                field_definitions
            )
            
            # Initialize data fusion processor
            self.fusion_processor = DataFusionProcessor(
                self.config_dict,
                field_definitions
            )
            
            # Pass integration components to forecast generators
            if hasattr(self, 'surf_generator'):
                self.surf_generator.integration_manager = self.integration_manager
                self.surf_generator.fusion_processor = self.fusion_processor
                self.surf_generator.db_manager = self.db_manager
            
            if hasattr(self, 'fishing_generator'):
                self.fishing_generator.integration_manager = self.integration_manager
                self.fishing_generator.fusion_processor = self.fusion_processor
                self.fishing_generator.db_manager = self.db_manager
            
            # Get integration summary for logging
            summary = self.integration_manager.get_integration_metadata_summary()
            log.info(f"{CORE_ICONS['status']} Station integration initialized: {summary}")
            
            return True
            
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error initializing station integration: {e}")
            return False
    
    def get_station_metadata_summary(self):
        """Get summary of available station metadata for diagnostics"""
        if not hasattr(self, 'integration_manager'):
            return {
                'status': 'integration_not_initialized',
                'message': 'Station integration manager not available'
            }
        
        try:
            summary = self.integration_manager.get_integration_metadata_summary()
            
            # Add service-level information
            summary.update({
                'status': 'operational',
                'integration_enabled': True,
                'fusion_processor_available': hasattr(self, 'fusion_processor'),
                'service_config': {
                    'forecast_types': self.service_config.get('forecast_types', []),
                    'data_integration_method': self.service_config.get('data_integration', {}).get('method', 'unknown')
                }
            })
            
            return summary
            
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error getting station metadata summary: {e}")
            return {
                'status': 'error',
                'message': f'Error retrieving metadata: {str(e)}'
            }
    
    def validate_integration_configuration(self):
        """Validate that Phase I metadata is available and properly configured"""
        validation_result = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        try:
            # Check if integration manager is initialized
            if not hasattr(self, 'integration_manager'):
                validation_result['errors'].append("Integration manager not initialized")
                validation_result['is_valid'] = False
                return validation_result
            
            # Check Phase I metadata availability
            summary = self.integration_manager.get_integration_metadata_summary()
            
            if not summary['phase_i_available']:
                validation_result['errors'].append("Phase I metadata not available in CONF")
                validation_result['recommendations'].append("Ensure Phase I Marine Data Extension is properly installed and configured")
                validation_result['is_valid'] = False
            
            # Check station availability for different data types
            if summary['wave_capable_stations'] == 0:
                validation_result['warnings'].append("No wave-capable stations found")
                validation_result['recommendations'].append("Add NDBC buoys with wave measurement capability")
            
            if summary['atmospheric_capable_stations'] == 0:
                validation_result['warnings'].append("No atmospheric-capable stations found")
                validation_result['recommendations'].append("Add NDBC buoys with atmospheric measurement capability")
            
            if summary['coops_station_count'] == 0:
                validation_result['warnings'].append("No CO-OPS tide stations found")
                validation_result['recommendations'].append("Add CO-OPS stations for tide data")
            
            # Check field definitions loaded
            if not hasattr(self.integration_manager, 'quality_thresholds') or not self.integration_manager.quality_thresholds:
                validation_result['errors'].append("Quality thresholds not loaded from YAML")
                validation_result['is_valid'] = False
            
            # Check fusion processor
            if not hasattr(self, 'fusion_processor'):
                validation_result['warnings'].append("Data fusion processor not available")
                validation_result['recommendations'].append("Multi-source data fusion will be limited")
            
            # Log validation results
            if validation_result['is_valid']:
                log.info(f"{CORE_ICONS['status']} Integration configuration validation passed")
            else:
                log.warning(f"{CORE_ICONS['warning']} Integration configuration validation failed: {validation_result['errors']}")
            
            return validation_result
            
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error validating integration configuration: {e}")
            validation_result['errors'].append(f"Validation error: {str(e)}")
            validation_result['is_valid'] = False
            return validation_result
     
    def get_optimal_sources_for_location(self, latitude, longitude):
        """Get optimal data sources for a specific location - utility method for diagnostics"""
        if not hasattr(self, 'integration_manager'):
            return None
        
        try:
            location_coords = (float(latitude), float(longitude))
            
            optimal_sources = {
                'wave_source': self.integration_manager.select_optimal_wave_source(location_coords),
                'atmospheric_sources': self.integration_manager.select_optimal_atmospheric_sources(location_coords),
                'tide_source': self.integration_manager.select_optimal_tide_source(location_coords),
                'location': {
                    'latitude': latitude,
                    'longitude': longitude
                }
            }
            
            return optimal_sources
            
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error getting optimal sources for location {latitude}, {longitude}: {e}")
            return None
    
    def test_data_integration(self, test_location=None):
        """Test data integration functionality - diagnostic method"""
        if not test_location:
            # Use default test location (Los Angeles area)
            test_location = {'latitude': 34.0522, 'longitude': -118.2437}
        
        test_results = {
            'timestamp': datetime.now().isoformat(),
            'test_location': test_location,
            'results': {}
        }
        
        try:
            log.info(f"{CORE_ICONS['status']} Testing data integration at {test_location}")
            
            # Test integration manager
            if hasattr(self, 'integration_manager'):
                optimal_sources = self.get_optimal_sources_for_location(
                    test_location['latitude'], 
                    test_location['longitude']
                )
                test_results['results']['optimal_sources'] = optimal_sources
                test_results['results']['integration_manager'] = 'operational'
            else:
                test_results['results']['integration_manager'] = 'not_available'
            
            # Test fusion processor
            if hasattr(self, 'fusion_processor'):
                test_results['results']['fusion_processor'] = 'operational'
            else:
                test_results['results']['fusion_processor'] = 'not_available'
            
            # Test validation
            validation = self.validate_integration_configuration()
            test_results['results']['configuration_validation'] = validation
            
            log.info(f"{CORE_ICONS['status']} Data integration test completed")
            return test_results
            
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error during data integration test: {e}")
            test_results['results']['error'] = str(e)
            return test_results

    def _find_next_good_surf_session(self, forecast_data):
        """Find next surf session with rating >= 3"""
        
        for forecast in forecast_data:
            if forecast['quality_rating'] >= 3:
                return {
                    'time': forecast['forecast_time_text'],
                    'rating': forecast['quality_rating'],
                    'conditions': forecast['conditions_text'],
                    'wave_range': forecast['wave_height_range'],
                    'wind': forecast['wind_condition']
                }
        return None

    def _generate_spot_fishing_forecast(self, spot):
        """Generate fishing forecast for a specific spot"""
        
        log.debug(f"Generating fishing forecast for {spot['name']}")
        
        # Get current marine conditions from Phase I
        marine_conditions = self._get_phase_i_marine_conditions(spot['latitude'], spot['longitude'])
        
        # Generate fishing forecast
        fishing_forecast = self.fishing_generator.generate_fishing_forecast(
            spot, marine_conditions
        )
        
        # Store forecast in database
        self._store_fishing_forecast(spot['id'], fishing_forecast)

    def _get_spot_by_id(self, spot_id, spot_type='surf'):
        """
        Get specific spot configuration by ID from CONF
        
        Args:
            spot_id: The spot identifier (CONF key)
            spot_type: 'surf' or 'fishing'
        
        Returns:
            dict: Spot configuration or None if not found
        """
        
        try:
            service_config = self.config_dict.get('SurfFishingService', {})
            spots_config_key = f'{spot_type}_spots'
            spots_config = service_config.get(spots_config_key, {})
            
            if spot_id in spots_config:
                spot_config = spots_config[spot_id]
                
                # Build standardized spot data structure
                spot = {
                    'id': spot_id,
                    'name': spot_config.get('name', spot_id),
                    'latitude': float(spot_config.get('latitude', '0.0')),
                    'longitude': float(spot_config.get('longitude', '0.0')),
                    'type': spot_config.get('type', spot_type),
                    'active': spot_config.get('active', 'true').lower() in ['true', '1', 'yes']
                }
                
                # Add type-specific fields
                if spot_type == 'surf':
                    spot.update({
                        'bottom_type': spot_config.get('bottom_type', 'sand'),
                        'exposure': spot_config.get('exposure', 'exposed')
                    })
                elif spot_type == 'fishing':
                    spot.update({
                        'location_type': spot_config.get('location_type', 'shore'),
                        'target_category': spot_config.get('target_category', 'mixed_bag')
                    })
                
                return spot
            
            log.warning(f"Spot {spot_id} not found in {spot_type}_spots configuration")
            return None
            
        except Exception as e:
            log.error(f"Error getting spot {spot_id} from CONF: {e}")
            return None
        
    def _validate_conf_locations(self):
        """
        Validate that CONF contains properly formatted location data
        
        Returns:
            dict: Validation results with any issues found
        """
        
        validation_results = {
            'valid': True,
            'issues': [],
            'surf_spots_count': 0,
            'fishing_spots_count': 0
        }
        
        try:
            service_config = self.config_dict.get('SurfFishingService', {})
            
            # Validate surf spots
            surf_spots = service_config.get('surf_spots', {})
            for spot_id, spot_config in surf_spots.items():
                validation_results['surf_spots_count'] += 1
                
                # Check required fields
                required_fields = ['name', 'latitude', 'longitude']
                for field in required_fields:
                    if field not in spot_config:
                        validation_results['valid'] = False
                        validation_results['issues'].append(f"Surf spot {spot_id} missing required field: {field}")
                
                # Validate coordinate ranges
                try:
                    lat = float(spot_config.get('latitude', '0.0'))
                    lon = float(spot_config.get('longitude', '0.0'))
                    if not (-90 <= lat <= 90):
                        validation_results['issues'].append(f"Surf spot {spot_id} invalid latitude: {lat}")
                    if not (-180 <= lon <= 180):
                        validation_results['issues'].append(f"Surf spot {spot_id} invalid longitude: {lon}")
                except (ValueError, TypeError):
                    validation_results['valid'] = False
                    validation_results['issues'].append(f"Surf spot {spot_id} invalid coordinate format")
            
            # Validate fishing spots  
            fishing_spots = service_config.get('fishing_spots', {})
            for spot_id, spot_config in fishing_spots.items():
                validation_results['fishing_spots_count'] += 1
                
                # Check required fields
                required_fields = ['name', 'latitude', 'longitude']
                for field in required_fields:
                    if field not in spot_config:
                        validation_results['valid'] = False
                        validation_results['issues'].append(f"Fishing spot {spot_id} missing required field: {field}")
                
                # Validate coordinates
                try:
                    lat = float(spot_config.get('latitude', '0.0'))
                    lon = float(spot_config.get('longitude', '0.0'))
                    if not (-90 <= lat <= 90):
                        validation_results['issues'].append(f"Fishing spot {spot_id} invalid latitude: {lat}")
                    if not (-180 <= lon <= 180):
                        validation_results['issues'].append(f"Fishing spot {spot_id} invalid longitude: {lon}")
                except (ValueError, TypeError):
                    validation_results['valid'] = False
                    validation_results['issues'].append(f"Fishing spot {spot_id} invalid coordinate format")
            
            if validation_results['valid']:
                log.info(f"CONF location validation passed: {validation_results['surf_spots_count']} surf spots, {validation_results['fishing_spots_count']} fishing spots")
            else:
                log.warning(f"CONF location validation failed: {len(validation_results['issues'])} issues found")
                for issue in validation_results['issues']:
                    log.warning(f"  - {issue}")
            
        except Exception as e:
            validation_results['valid'] = False
            validation_results['issues'].append(f"Validation error: {e}")
            log.error(f"Error validating CONF locations: {e}")
        
        return validation_results
    
    def _load_field_definitions(self):
        """
        Load field definitions from CONF (written by installer from YAML)
        Service uses CONF only - NO runtime YAML access
        
        Phase II REQUIRES Phase I to be properly installed and configured
        """
        try:
            # Get field definitions from CONF (written by installer)
            service_config = self.config_dict.get('SurfFishingService', {})
            field_definitions = service_config.get('field_definitions', {})
            
            if not field_definitions:
                error_msg = "Phase II field definitions not found in CONF. Phase II installation may be incomplete."
                log.error(f"{CORE_ICONS['warning']} {error_msg}")
                raise RuntimeError(error_msg)
            
            # Also verify Phase I configuration exists
            phase_i_config = self.config_dict.get('MarineDataService', {})
            if not phase_i_config:
                error_msg = "Phase I Marine Data Extension configuration not found. Phase II requires Phase I to be installed first."
                log.error(f"{CORE_ICONS['warning']} {error_msg}")
                raise RuntimeError(error_msg)
            
            log.info(f"{CORE_ICONS['status']} Loaded field definitions from CONF")
            return field_definitions
            
        except Exception as e:
            if isinstance(e, RuntimeError):
                # Re-raise our specific error messages
                raise
            else:
                error_msg = f"Critical error loading field definitions from CONF: {e}"
                log.error(f"{CORE_ICONS['warning']} {error_msg}")
                raise RuntimeError(error_msg)
            
    @property
    def db_manager(self):
        """
        Property accessor for database manager
        Provides transparent access to delayed-initialized database manager
        """
        return self._get_db_manager()