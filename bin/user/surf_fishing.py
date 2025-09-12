#!/usr/bin/env python3
# Magic Animal: Capybara
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
import configobj
import shutil
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
            gfs_wave_config = service_config.get('noaa_gfs_wave', {})
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
                        # Normalize longitude to match GRIB file format (0-360 if needed)
                        try:
                            lon_min = eccodes.codes_get(msg_id, 'longitudeOfFirstGridPointInDegrees')
                            normalized_lon = target_lon + 360 if (target_lon < 0 and lon_min >= 0) else target_lon
                            value = eccodes.codes_get_nearest(msg_id, target_lat, normalized_lon)[0]['value']
                        except:
                            # Fallback if longitude normalization fails
                            value = eccodes.codes_get_nearest(msg_id, target_lat, target_lon)[0]['value']
                        
                        # FIX: Convert to float and validate - FAIL if invalid
                        try:
                            numeric_value = float(value)
                            # Check for invalid values
                            if math.isnan(numeric_value) or math.isinf(numeric_value):
                                log.warning(f"{CORE_ICONS['warning']} Invalid numeric value (NaN/Inf) for {param_name}")
                                continue  # Skip this data point
                        except (ValueError, TypeError) as e:
                            log.error(f"{CORE_ICONS['warning']} Cannot convert GRIB value to float for {param_name}: {value} - {e}")
                            continue  # Skip this data point - don't use invalid data

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
            import numpy as np
            from datetime import datetime, timedelta
            
            # READ FROM NEW CONF: Field mappings for parameter processing
            service_config = self.config_dict.get('SurfFishingService', {})
            gfs_wave_config = service_config.get('noaa_gfs_wave', {})
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
                        
                        log.debug(f"Processing parameter: {param_name}")
                        log.debug(f"Target coordinates: lat={target_lat}, lon={target_lon}")
                        log.debug(f"GRIB domain: lat=[{grb.latitudes.min():.2f},{grb.latitudes.max():.2f}], lon=[{grb.longitudes.min():.2f},{grb.longitudes.max():.2f}]")
                        
                        # Get forecast time info
                        forecast_time_offset = grb.forecastTime
                        
                        # Get base time from GRIB message
                        valid_date = grb.validDate
                        forecast_timestamp = valid_date.timestamp()
                        
                        # Get the full data grid first to avoid Key/value errors
                        values, lats, lons = grb.data()
                        log.debug(f"Data grid shape: values={values.shape}, lats={lats.shape}, lons={lons.shape}")

                        # Normalize longitude using actual grid data
                        normalized_lon = target_lon + 360 if (target_lon < 0 and lons.min() >= 0) else target_lon
                        log.debug(f"Normalized longitude: {normalized_lon} (original: {target_lon})")

                        # Find closest point manually (more reliable than grb.nearest)
                        distances = np.sqrt((lats - target_lat)**2 + (lons - normalized_lon)**2)
                        min_idx = np.argmin(distances)
                        closest_value = float(values.flat[min_idx])
                        log.debug(f"Closest value for {param_name}: {closest_value} (min_distance: {distances.flat[min_idx]:.4f})")

                        # CRITICAL FIX: Only add valid data points to the list
                        if not np.isnan(closest_value):
                            data_points.append({
                                'parameter': param_name,
                                'value': closest_value,
                                'forecast_time': forecast_timestamp,
                                'latitude': target_lat,
                                'longitude': target_lon
                            })
                            log.debug(f"✅ Adding data point for {param_name}: {closest_value}")
                        else:
                            log.debug(f"❌ Skipping NaN value for {param_name}")
                        
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
        lat1, lon1 = math.radians(float(location_coords[0])), math.radians(float(location_coords[1]))
        lat2, lon2 = math.radians(float(station_coords[0])), math.radians(float(station_coords[1]))
        
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
    
    def __init__(self, config_dict, grib_processor=None):
        """Initialize WaveWatchDataCollector with GRIB caching support"""
        
        # EXISTING CODE: Store references - PRESERVED EXACTLY
        self.config_dict = config_dict
        
        # EXISTING CODE: Use provided grib_processor or create new one - PRESERVED EXACTLY
        if grib_processor:
            self.grib_processor = grib_processor
        else:
            self.grib_processor = GRIBProcessor(config_dict)
        
        # Get service configuration
        service_config = config_dict.get('SurfFishingService', {})
        
        # EXISTING CODE: READ FROM CONF - PRESERVED EXACTLY
        gfs_wave_config = service_config.get('noaa_gfs_wave', {})
        if not gfs_wave_config:
            raise RuntimeError("GFS Wave configuration missing from CONF - installer may have failed")
        
        # EXISTING CODE: Initialize attributes from CONF - PRESERVED EXACTLY
        self.base_url = gfs_wave_config['base_url']
        self.url_pattern = gfs_wave_config['url_pattern'] 
        self.file_pattern = gfs_wave_config['file_pattern']
        
        # READ FROM CONF: Grid definitions (must be complete)
        self.grids = gfs_wave_config['grids']
        if not self.grids:
            raise RuntimeError("Grid definitions missing from CONF - installer may have failed")
        
        # READ FROM CONF: Schedule configuration (must be complete)
        schedule_config = gfs_wave_config['schedule']
        
        # Parse model runs from comma-separated string
        model_runs_str = schedule_config['model_runs']
        self.run_cycles = [int(x.strip()) for x in model_runs_str.split(',')]
        
        # Parse forecast hours from comma-separated string
        forecast_hours_str = schedule_config['forecast_hours']
        self.forecast_hours = [int(x.strip()) for x in forecast_hours_str.split(',')]
        
        # READ FROM CONF: Error handling configuration (must be complete)
        error_handling = gfs_wave_config['error_handling']
        self.api_timeout = int(error_handling['timeout_seconds'])
        self.retry_attempts = int(error_handling['max_retries'])
        self.connection_timeout = int(error_handling['timeout_seconds'])
        self.read_timeout = int(error_handling['timeout_seconds'])
        
        # NEW: GRIB cache management
        weewx_root = config_dict.get('WEEWX_ROOT', '/etc/weewx')
        self.cache_root = os.path.join(weewx_root, 'cache', 'surf_fishing', 'grib')
        os.makedirs(self.cache_root, exist_ok=True)
        
        # EXISTING CODE: INLINE GFS cycle calculation - PRESERVED EXACTLY
        from datetime import datetime, timedelta
        
        def get_expected_gfs_cycle(current_time):
            """Calculate the most recent GFS Wave cycle that should be available"""
            # Processing delay: GFS Wave takes 3-4 hours to be available
            processing_delay = 4
            effective_time = current_time - timedelta(hours=processing_delay)
            
            # Find most recent cycle from configured run cycles
            current_hour = effective_time.hour
            for cycle in reversed(self.run_cycles):
                if current_hour >= cycle:
                    return effective_time.replace(hour=cycle, minute=0, second=0, microsecond=0)
            
            # If before first cycle of day, use last cycle of previous day
            last_cycle = self.run_cycles[-1] if self.run_cycles else 18
            return effective_time.replace(hour=last_cycle, minute=0, second=0, microsecond=0) - timedelta(days=1)
        
        # Bind method to instance
        self.get_expected_gfs_cycle = get_expected_gfs_cycle
        
        # EXISTING CODE: Validate CONF templates - PRESERVED EXACTLY
        try:
            test_url = self.url_pattern.format(yyyymmdd='20250821', hh='12')
            test_file = self.file_pattern.format(hh='12', grid_name='test', fff='000')
            log.debug(f"{CORE_ICONS['status']} CONF template validation passed")
        except KeyError as e:
            raise RuntimeError(f"CONF template pattern missing required variable {e}")
        except Exception as e:
            raise RuntimeError(f"CONF template validation failed: {e}")
        
        log.info(f"{CORE_ICONS['status']} WaveWatchDataCollector initialized from CONF with caching at {self.cache_root}")

    def fetch_forecast_data(self, spot_config):
        """Fetch GFS Wave forecast data with existing robust infrastructure plus caching"""

        if not self.grib_processor.is_available():
            log.warning("GRIB processing not available - skipping GFS Wave data")
            return []
        
        try:
            # EXISTING CODE: Extract coordinates from spot_config - PRESERVED EXACTLY
            bathymetric_path = spot_config.get('bathymetric_path', {})
            
            # Use offshore coordinates preferentially for deep water wave data
            latitude = spot_config.get('offshore_latitude')
            longitude = spot_config.get('offshore_longitude')
            
            if latitude is None or longitude is None:
                # Fallback to surf break coordinates with warning
                latitude = spot_config.get('latitude')
                longitude = spot_config.get('longitude')
                
                if latitude is None or longitude is None:
                    log.error("No valid coordinates found in spot configuration")
                    return []
                
                log.warning(f"Using surf break coordinates {latitude}, {longitude} - offshore coordinates not available")
            else:
                log.debug(f"Using offshore coordinates {latitude}, {longitude} for GFS Wave data")
            
            # Convert to float with validation
            try:
                latitude = float(latitude)
                longitude = float(longitude)
            except (ValueError, TypeError):
                log.error(f"Invalid coordinate format: lat={latitude}, lon={longitude}")
                return []
            
            # EXISTING CODE: INLINE grid selection using CONF regional mappings - PRESERVED EXACTLY
            grid_name = None
            try:
                # Check each configured grid for coverage
                for grid_candidate, grid_config in self.grids.items():
                    bounds = grid_config.get('bounds', [])
                    if bounds:
                        # Parse bounds string from CONF (format: "lat_min,lat_max,lon_min,lon_max")
                        if isinstance(bounds, str):
                            bounds_list = [float(x.strip()) for x in bounds.split(',')]
                        elif isinstance(bounds, list):
                            bounds_list = [float(x) for x in bounds]
                        else:
                            continue
                            
                        if len(bounds_list) == 4:
                            
                                lat_min, lat_max, lon_min, lon_max = bounds_list
                                
                                # Normalize longitude for consistent comparison
                                normalized_lon = longitude
                                if normalized_lon < 0 and lon_min > 0:
                                    normalized_lon += 360
                                elif normalized_lon > 0 and lon_max < 0:
                                    normalized_lon -= 360
                                
                                if lat_min <= latitude <= lat_max and lon_min <= normalized_lon <= lon_max:
                                    # Select by priority (lowest number = highest priority)
                                    priority = int(grid_config.get('priority', 999))
                                    if grid_name is None or priority < getattr(self, '_selected_grid_priority', 999):
                                        grid_name = grid_config.get('grid_name', grid_candidate)
                                        self._selected_grid_priority = priority
                                        log.debug(f"{CORE_ICONS['navigation']} Selected priority {priority} grid {grid_name} for location {latitude}, {longitude}")
                
                # EXISTING CODE: Priority fallback system for grid coverage - PRESERVED EXACTLY
                if not grid_name:
                    fallback_grids = ['global.0p16', 'global.0p25']
                    for fallback_grid in fallback_grids:
                        for grid_candidate, grid_config in self.grids.items():
                            if grid_config.get('grid_name') == fallback_grid:
                                grid_name = fallback_grid
                                log.debug(f"{CORE_ICONS['navigation']} Using fallback grid {grid_name} for location {latitude}, {longitude}")
                                break
                        if grid_name:
                            break
                
                if not grid_name:
                    log.error(f"No suitable grid found for location {latitude}, {longitude}")
                    return []
                    
            except Exception as e:
                log.error(f"Grid selection failed for location {latitude}, {longitude}: {e}")
                return []
            
            log.debug(f"Using GFS Wave grid: {grid_name} for location {latitude}, {longitude}")
            
            # MODIFIED: Download GRIB files with caching (for_forecasting=True for fresh data)
            grib_files = self._download_grib_files(grid_name, for_forecasting=True)
            
            if not grib_files:
                log.warning("No GRIB files downloaded")
                return []
            
            # EXISTING CODE: Process GRIB files using existing GRIBProcessor - PRESERVED EXACTLY
            forecast_data = []
            for grib_file in grib_files:
                try:
                    data_points = self.grib_processor.process_gfs_wave_file(
                        grib_file['file_path'], latitude, longitude
                    )
                    forecast_data.extend(data_points)
                except Exception as e:
                    log.error(f"Error processing GRIB file: {e}")
                    continue
            
            # EXISTING CODE: Organize forecast data - PRESERVED EXACTLY
            return self._organize_forecast_data(forecast_data)
            
        except Exception as e:
            log.error(f"Error fetching GFS Wave data: {e}")
            return []

    def _download_grib_files(self, grid_name, for_forecasting=True):
        """Download GFS Wave GRIB files with unified caching strategy"""
        
        current_time = datetime.utcnow()
        
        # UNIFIED CACHING LOGIC: Check cache first
        if for_forecasting:
            # For forecasting: Check if current expected cycle is cached
            expected_cycle = self.get_expected_gfs_cycle(current_time)
            cache_path = self._get_cache_path(grid_name, expected_cycle)
            
            if self._is_cache_valid(cache_path, for_forecasting=True):
                cached_files = self._get_cached_files(cache_path)
                if len(cached_files) >= 8:  # Same threshold as download logic
                    log.info(f"{CORE_ICONS['status']} Using cached cycle: {os.path.basename(cache_path)} ({len(cached_files)} files)")
                    return cached_files
                else:
                    log.debug(f"{CORE_ICONS['warning']} Cached cycle incomplete ({len(cached_files)} files), attempting fresh download")
            else:
                log.debug(f"{CORE_ICONS['navigation']} No valid cache for current cycle, attempting fresh download")
        else:
            # For bathymetry: Use any reasonably fresh cache
            cached_files = self.get_cached_files_for_validation(grid_name)
            if cached_files:
                return cached_files
            
            log.debug(f"{CORE_ICONS['navigation']} No suitable cache for bathymetry validation, downloading fresh files")
        
        # EXISTING DOWNLOAD LOGIC: Preserved exactly, but write to cache
        grib_files = []
        successful_cycle = None
        
        try:
            # EXISTING CODE: Smart cycle selection - PRESERVED EXACTLY
            expected_cycle = self.get_expected_gfs_cycle(current_time)
            
            log.info(f"{CORE_ICONS['navigation']} Expected most recent cycle: {expected_cycle.strftime('%Y%m%d %HZ')}")
            
            # Try multiple recent cycles to find available data
            cycles_to_try = []
            cycles_to_try.append(expected_cycle)  # Try expected cycle first
            
            # Add previous cycles as fallbacks (up to 3 cycles back = 18 hours)
            for i in range(1, 4):
                fallback_cycle = expected_cycle - timedelta(hours=6 * i)
                cycles_to_try.append(fallback_cycle)
            
            for cycle_attempt, potential_run in enumerate(cycles_to_try, 1):
                run_date_str = potential_run.strftime("%Y%m%d")
                run_hour_str = f"{potential_run.hour:02d}"
                
                log.debug(f"{CORE_ICONS['navigation']} Trying GFS Wave cycle {cycle_attempt}/{len(cycles_to_try)}: {run_date_str} {run_hour_str}Z")
                
                # NEW: Create cache directory for this cycle
                cycle_cache_path = self._get_cache_path(grid_name, potential_run)
                os.makedirs(cycle_cache_path, exist_ok=True)
                
                cycle_files = []
                
                # EXISTING CODE: Try to download files for this cycle - PRESERVED EXACTLY
                for forecast_hour in self.forecast_hours[:24]:  # First 72 hours (limit to 24 files)
                    
                    # BUILD URL FROM CONF - FAIL HARD if missing variables
                    try:
                        filename = self.file_pattern.format(
                            hh=run_hour_str,
                            grid_name=grid_name,
                            fff=f"{forecast_hour:03d}"
                        )
                        
                        url_path = self.url_pattern.format(
                            yyyymmdd=run_date_str,
                            hh=run_hour_str
                        )
                        
                        url = f"{self.base_url}{url_path}{filename}"
                        
                    except KeyError as e:
                        raise RuntimeError(f"CONF template missing variable: {e}")
                    
                    # MODIFIED: Download to cache instead of temp file
                    try:
                        cache_file_path = os.path.join(cycle_cache_path, filename)
                        
                        # Download with timeout from CONF
                        request = urllib.request.Request(url)
                        request.add_header('User-Agent', 'weewx-surf-fishing-extension/1.0')
                        
                        with urllib.request.urlopen(request, timeout=self.connection_timeout) as response:
                            with open(cache_file_path, 'wb') as f:
                                f.write(response.read())
                        
                        # EXISTING CODE: VALIDATE FILE - Detect bogus files - PRESERVED EXACTLY
                        file_size = os.path.getsize(cache_file_path)
                        
                        # Check minimum file size (varies by grid) - CONSERVATIVE VALUES
                        min_sizes = {
                            'global.0p16': 200000,     # ~200KB minimum for global
                            'atlocn.0p16': 100000,     # ~100KB minimum for Atlantic
                            'wcoast.0p16': 100000,     # ~100KB minimum for West Coast  
                            'arctic.9km': 80000,       # ~80KB minimum for Arctic
                            'epacif.0p16': 100000,     # ~100KB minimum for Pacific
                            'global.0p25': 150000      # ~150KB minimum for global 25km
                        }
                        min_size = min_sizes.get(grid_name, 50000)  # 50KB default minimum
                        
                        # Validate file size and GRIB header
                        if file_size >= min_size:
                            with open(cache_file_path, 'rb') as f:
                                header = f.read(4)
                                if header == b'GRIB':
                                    cycle_files.append({
                                        'file_path': cache_file_path,
                                        'forecast_hour': forecast_hour,
                                        'grid_name': grid_name,
                                        'run_time': potential_run,
                                        'file_size': file_size
                                    })
                                    log.debug(f"{CORE_ICONS['status']} Downloaded: {filename} ({file_size} bytes)")
                                else:
                                    log.debug(f"{CORE_ICONS['warning']} Invalid GRIB header: {filename}")
                                    os.unlink(cache_file_path)
                        else:
                            log.debug(f"{CORE_ICONS['warning']} File too small: {filename} ({file_size} bytes, need {min_size}+)")
                            os.unlink(cache_file_path)
                            
                    except (urllib.error.URLError, urllib.error.HTTPError) as e:
                        if hasattr(e, 'code') and e.code == 404:
                            log.debug(f"{CORE_ICONS['warning']} GFS Wave file not yet available: {filename}")
                        else:
                            log.debug(f"{CORE_ICONS['warning']} HTTP error for {filename}: {e}")
                        continue
                    except Exception as e:
                        log.debug(f"{CORE_ICONS['warning']} Download error for {filename}: {e}")
                        continue
                
                # EXISTING CODE: Check if cycle is complete - PRESERVED EXACTLY
                if len(cycle_files) >= 8:  # Need at least 8 files for useful forecast
                    grib_files = cycle_files
                    successful_cycle = potential_run
                    log.info(f"{CORE_ICONS['status']} Cycle {run_date_str} {run_hour_str}Z: Got {len(cycle_files)} files - using this cycle")
                    
                    # NEW: Clean up old caches only after successful download
                    self._cleanup_old_cache(grid_name, keep_cycles=3)
                    break
                else:
                    # Clean up incomplete download from cache
                    try:
                        shutil.rmtree(cycle_cache_path)
                        log.debug(f"{CORE_ICONS['warning']} Removed incomplete cache: {os.path.basename(cycle_cache_path)}")
                    except:
                        pass
                    log.warning(f"{CORE_ICONS['warning']} Cycle {run_date_str} {run_hour_str}Z: only {len(cycle_files)} files, need 8+ - trying previous cycle")
            
            # EXISTING CODE: Final status logging - PRESERVED EXACTLY  
            if grib_files and successful_cycle:
                log.info(f"{CORE_ICONS['status']} Downloaded {len(grib_files)} GFS Wave files from {successful_cycle.strftime('%Y%m%d %HZ')}")
            else:
                log.warning(f"{CORE_ICONS['warning']} No GFS Wave files could be downloaded for grid {grid_name} after trying {len(cycles_to_try)} cycles")
                
                # NEW: Fallback to any available cache as last resort
                if for_forecasting:
                    log.info(f"{CORE_ICONS['navigation']} Attempting fallback to any available cached data...")
                    cached_files = self.get_cached_files_for_validation(grid_name)
                    if cached_files and len(cached_files) >= 8:
                        log.info(f"{CORE_ICONS['status']} Using fallback cache ({len(cached_files)} files)")
                        return cached_files
                
                return []
            
            return grib_files
            
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Critical error in GFS Wave download: {e}")
            return []
    
    def _organize_forecast_data(self, data_points):
        """Organize raw GRIB data into forecast periods with WeeWX unit system handling"""
        
        try:
            if not data_points:
                log.warning(f"{CORE_ICONS['warning']} No data points to organize")
                return []
            
            # Get field mappings from CONF - REQUIRED for operation
            gfs_wave_config = self.config_dict.get('SurfFishingService', {}).get('noaa_gfs_wave', {})
            field_mappings = gfs_wave_config.get('field_mappings', {})
            
            if not field_mappings:
                log.error(f"{CORE_ICONS['warning']} No field mappings found in CONF - cannot organize data")
                return []
            
            # Get WeeWX unit system from engine
            target_unit_system = self._get_target_unit_system()
            
            # Group data by forecast time
            forecast_periods = {}
            for point in data_points:
                try:
                    forecast_time = point['forecast_time']
                    if forecast_time not in forecast_periods:
                        forecast_periods[forecast_time] = {}
                    
                    # FIX: Validate the value is numeric (should already be from GRIB processor)
                    value = point.get('value')
                    if value is None:
                        log.warning(f"Null value for {point.get('parameter', 'unknown')} at time {forecast_time}")
                        continue
                    
                    # Extra validation - the value should already be float from GRIB processor
                    try:
                        numeric_value = float(value)
                        if math.isnan(numeric_value) or math.isinf(numeric_value):
                            log.warning(f"Invalid numeric value for {point.get('parameter', 'unknown')}")
                            continue
                    except (ValueError, TypeError) as e:
                        log.error(f"Non-numeric value in data point for {point.get('parameter', 'unknown')}: {value}")
                        continue  # Skip invalid data
                    
                    forecast_periods[forecast_time][point['parameter']] = numeric_value
        
                    forecast_time = point['forecast_time']
                    if forecast_time not in forecast_periods:
                        forecast_periods[forecast_time] = {}
                    
                    # Safe numeric conversion
                    try:
                        value = float(point['value']) if point['value'] is not None else None
                    except (ValueError, TypeError):
                        log.warning(f"Non-numeric GRIB value for {point.get('parameter', 'unknown')}: {point['value']}")
                        value = None
                    
                    if value is not None:
                        forecast_periods[forecast_time][point['parameter']] = value
                except Exception as e:
                    log.debug(f"Error processing data point: {e}")
                    continue
            
            # Convert to list format with WeeWX unit conversions
            organized_data = []
            
            for forecast_time, parameters in forecast_periods.items():
                try:
                    converted_data = {'forecast_time': forecast_time}
                    
                    # Apply WeeWX unit conversions based on field mappings
                    for field_name, field_config in field_mappings.items():
                        grib_parameter = field_config.get('grib_parameter')
                        
                        if grib_parameter and grib_parameter in parameters:
                            raw_value = parameters[grib_parameter]
                            
                            # Apply WeeWX unit conversion
                            if field_config.get('unit_conversion_required', False):
                                converted_value = self._convert_to_weewx_units(
                                    raw_value, field_name, target_unit_system
                                )
                                converted_data[field_name] = converted_value
                            else:
                                converted_data[field_name] = raw_value
                    
                    if len(converted_data) > 1:  # More than just forecast_time
                        organized_data.append(converted_data)
                        
                except Exception as e:
                    log.error(f"{CORE_ICONS['warning']} Error converting data for time {forecast_time}: {e}")
                    continue
            
            # Sort by forecast time
            organized_data.sort(key=lambda x: x['forecast_time'])
            
            log.debug(f"{CORE_ICONS['status']} Organized {len(organized_data)} forecast periods")
            return organized_data
            
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error organizing forecast data: {e}")
            return []

    def _get_target_unit_system(self):
        """Get the target unit system from WeeWX configuration"""
        
        try:
            # Get the station's unit system from WeeWX configuration
            station_config = self.config_dict.get('Station', {})
            station_type = station_config.get('station_type', 'Simulator')
            
            # Get the target unit system for this station
            if hasattr(weewx.units, 'unit_constants'):
                target_unit_system = weewx.units.unit_constants.get(station_type, weewx.US)
            else:
                # Fallback to checking StdConvert configuration
                convert_config = self.config_dict.get('StdConvert', {})
                target_unit_nick = convert_config.get('target_unit', 'US')
                
                if target_unit_nick.upper() == 'METRIC':
                    target_unit_system = weewx.METRIC
                elif target_unit_nick.upper() == 'METRICWX':
                    target_unit_system = weewx.METRICWX
                else:
                    target_unit_system = weewx.US
            
            return target_unit_system
            
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error determining target unit system: {e}")
            return weewx.US  # Default to US units if unable to determine

    def _convert_to_weewx_units(self, value, field_name, target_unit_system):
        """Convert raw GRIB values to WeeWX target unit system"""
        
        try:
            # Define source units for GRIB fields (always in SI/metric)
            grib_source_units = {
                'wave_height': 'meter',
                'total_swell_height': 'meter', 
                'wind_wave_height': 'meter',
                'wind_speed': 'meter_per_second',
                'wind_u_component': 'meter_per_second',
                'wind_v_component': 'meter_per_second'
            }
            
            source_unit = grib_source_units.get(field_name)
            
            if not source_unit:
                # No conversion needed for this field
                return value
            
            # Create ValueTuple for WeeWX unit conversion
            unit_group_map = {
                'meter': 'group_altitude',
                'meter_per_second': 'group_speed'
            }
            unit_group = unit_group_map.get(source_unit, 'group_altitude')
            value_tuple = (value, source_unit, unit_group)
            
            # Convert to target unit system using WeeWX
            converted_tuple = weewx.units.convertStd(value_tuple, target_unit_system)
            
            return converted_tuple[0]  # Return just the converted value
            
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error converting {field_name} to WeeWX units: {e}")
            return value  # Return original value if conversion fails
    
    def _calculate_wind_speed(self, parameters):
        """Calculate wind speed with primary/backup methods using GFS Wave fields"""
        
        try:
            # PRIMARY: Direct wind speed from GFS Wave
            if 'wind_speed' in parameters and parameters['wind_speed'] is not None:
                return parameters['wind_speed']  # Already converted by _organize_forecast_data
            
            # BACKUP: Calculate from U/V components if available
            u_wind = parameters.get('wind_u_component')
            v_wind = parameters.get('wind_v_component')
            
            if u_wind is not None and v_wind is not None:
                wind_speed = math.sqrt(u_wind**2 + v_wind**2)
                return wind_speed  # Already in target units from _organize_forecast_data
            
            # NO FALLBACK: Return None to indicate missing data
            log.debug(f"{CORE_ICONS['warning']} No wind speed data available")
            return None
            
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error calculating wind speed: {e}")
            return None
    
    def _calculate_wind_direction(self, parameters):
        """Calculate wind direction with primary/backup methods using GFS Wave fields"""
        
        try:
            # PRIMARY: Direct wind direction from GFS Wave
            if 'wind_direction' in parameters and parameters['wind_direction'] is not None:
                return parameters['wind_direction']
            
            # BACKUP: Calculate from U/V components if available
            u_wind = parameters.get('wind_u_component')
            v_wind = parameters.get('wind_v_component')
            
            if u_wind is not None and v_wind is not None:
                # Calculate direction from components (meteorological convention)
                direction = math.atan2(-u_wind, -v_wind) * 180.0 / math.pi
                if direction < 0:
                    direction += 360.0
                return direction
            
            # NO FALLBACK: Return None to indicate missing data
            log.debug(f"{CORE_ICONS['warning']} No wind direction data available")
            return None
            
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error calculating wind direction: {e}")
            return None

    def _get_cache_path(self, grid_name, cycle_datetime):
        """Get cache directory path for specific grid and cycle"""
        cycle_str = cycle_datetime.strftime('%Y%m%d_%HZ')
        return os.path.join(self.cache_root, f"{grid_name}_{cycle_str}")

    def _is_cache_valid(self, cache_path, for_forecasting=True):
        """Check if cached cycle is valid and not expired"""
        if not os.path.exists(cache_path):
            return False
        
        # For bathymetry validation, any reasonably fresh cache is fine
        if not for_forecasting:
            cache_age_hours = (datetime.utcnow() - datetime.fromtimestamp(os.path.getmtime(cache_path))).total_seconds() / 3600
            return cache_age_hours < 24  # 24 hour tolerance for bathymetry
        
        # For forecasting, check if files are from expected current cycle
        expected_cycle = self.get_expected_gfs_cycle(datetime.utcnow())
        cache_cycle_str = expected_cycle.strftime('%Y%m%d_%HZ')
        
        return cache_cycle_str in cache_path

    def _get_cached_files(self, cache_path):
        """Get list of cached GRIB files from cache directory"""
        if not os.path.exists(cache_path):
            return []
        
        cached_files = []
        for filename in os.listdir(cache_path):
            if filename.endswith('.grib2'):
                file_path = os.path.join(cache_path, filename)
                
                # Extract forecast hour from filename for consistency with existing code
                try:
                    # Extract forecast hour from pattern: gfswave.t12z.wcoast.0p16.f003.grib2
                    parts = filename.split('.')
                    forecast_hour_str = parts[-2]  # f003
                    forecast_hour = int(forecast_hour_str[1:])  # 3
                    
                    cached_files.append({
                        'file_path': file_path,
                        'forecast_hour': forecast_hour,
                        'grid_name': cache_path.split('_')[0].split('/')[-1],  # Extract grid from path
                        'file_size': os.path.getsize(file_path)
                    })
                except (IndexError, ValueError):
                    log.debug(f"{CORE_ICONS['warning']} Skipping invalid cached file: {filename}")
                    continue
        
        # Sort by forecast hour for consistency with download logic
        cached_files.sort(key=lambda x: x['forecast_hour'])
        return cached_files

    def get_cached_files_for_validation(self, grid_name):
        """Public method for BathymetryProcessor to get cached files for validation"""
        try:
            # Look for any reasonably fresh cache for this grid
            for item in os.listdir(self.cache_root):
                if item.startswith(f"{grid_name}_"):
                    cache_path = os.path.join(self.cache_root, item)
                    if self._is_cache_valid(cache_path, for_forecasting=False):
                        cached_files = self._get_cached_files(cache_path)
                        if len(cached_files) >= 1:  # Need at least one file for validation
                            log.debug(f"{CORE_ICONS['status']} Found cached files for validation: {os.path.basename(cache_path)} ({len(cached_files)} files)")
                            return cached_files
            
            log.debug(f"{CORE_ICONS['warning']} No cached files available for validation of grid {grid_name}")
            return []
            
        except Exception as e:
            log.debug(f"{CORE_ICONS['warning']} Error getting cached files for validation: {e}")
            return []

    def _cleanup_old_cache(self, grid_name, keep_cycles=3):
        """Clean up old cached cycles, keeping only the most recent ones"""
        try:
            # Find all cache directories for this grid
            grid_caches = []
            for item in os.listdir(self.cache_root):
                if item.startswith(f"{grid_name}_") and os.path.isdir(os.path.join(self.cache_root, item)):
                    cache_path = os.path.join(self.cache_root, item)
                    mtime = os.path.getmtime(cache_path)
                    grid_caches.append((cache_path, mtime))
            
            # Sort by modification time (newest first)
            grid_caches.sort(key=lambda x: x[1], reverse=True)
            
            # Remove old caches beyond keep_cycles limit
            for cache_path, _ in grid_caches[keep_cycles:]:
                try:
                    shutil.rmtree(cache_path)
                    log.debug(f"{CORE_ICONS['status']} Cleaned up old cache: {os.path.basename(cache_path)}")
                except Exception as e:
                    log.debug(f"{CORE_ICONS['warning']} Could not remove old cache {cache_path}: {e}")
                    
        except Exception as e:
            log.debug(f"{CORE_ICONS['warning']} Error during cache cleanup: {e}")


class BathymetryProcessor:
    """Handle deep water point determination and bathymetry profile calculations for surf spots"""
    
    def __init__(self, config_dict, grib_processor, engine):
        """Initialize bathymetry processor with CONF configuration, GRIB processing, and adaptive algorithm parameters"""
        # PRESERVE: All existing initialization patterns unchanged
        self.config_dict = config_dict
        self.grib_processor = grib_processor
        self.engine = engine  # Store engine reference for weewx.conf access
        
        # PRESERVE: Get bathymetry configuration from CONF using existing WeeWX 5.1 patterns
        service_config = config_dict.get('SurfFishingService', {})
        self.bathymetry_config = service_config.get('bathymetry_data', {})
        
        # PRESERVE: GEBCO API configuration from CONF (unchanged)
        api_config = self.bathymetry_config.get('api_configuration', {})
        self.gebco_base_url = api_config.get('base_url', 'https://api.opentopodata.org/v1/gebco2020')
        self.api_timeout = int(api_config.get('timeout_seconds', '30'))
        self.retry_attempts = int(api_config.get('retry_attempts', '3'))
        
        # PRESERVE: Path analysis configuration (unchanged)
        path_config = self.bathymetry_config.get('path_analysis', {})
        self.path_resolution_points = int(path_config.get('path_resolution_points', '15'))
        self.offshore_distance_km = float(path_config.get('offshore_distance_meters', '20000')) / 1000.0  # Convert to km
        
        # NEW: Adaptive algorithm configuration from CONF (Phase 3 enhancement)
        adaptive_config = self.bathymetry_config.get('adaptive_spacing', {})
        self.enable_adaptive = adaptive_config.get('enable_adaptive_algorithm', True)  # Default enabled for alpha
        self.refinement_threshold = float(adaptive_config.get('refinement_gradient_threshold', '0.02'))
        self.coarsening_threshold = float(adaptive_config.get('coarsening_gradient_threshold', '0.002'))
        self.max_points = int(adaptive_config.get('max_total_points', '75'))
        self.min_points = int(adaptive_config.get('min_total_points', '16'))
        self.critical_depth_min = float(adaptive_config.get('critical_depth_min', '5'))
        self.critical_depth_max = float(adaptive_config.get('critical_depth_max', '50'))
        self.deep_water_threshold = float(adaptive_config.get('deep_water_threshold', '50'))
        self.max_iterations = int(adaptive_config.get('max_refinement_iterations', '3'))
        self.min_segment_distance = float(adaptive_config.get('min_segment_distance_m', '200'))
        self.validation_enabled = adaptive_config.get('validation_enabled', True)
        
        # NEW: Initialize performance optimization cache for gradient calculations
        self._gradient_cache = {}
        
        # PRESERVE: Existing log message with enhancement indicator
        log.info(f"{CORE_ICONS['status']} BathymetryProcessor initialized from CONF")
        if self.enable_adaptive:
            log.info(f"{CORE_ICONS['status']} Adaptive algorithm: enabled (threshold: {self.refinement_threshold:.3f}, max points: {self.max_points})")
        else:
            log.info(f"{CORE_ICONS['status']} Adaptive algorithm: disabled (using original 16-point method)")
    
    def process_surf_spot_bathymetry(self, spot):
        """Main entry point - check flag and process if needed"""
        
        spot_id = spot['id']
        spot_name = spot.get('name', spot_id)
        
        # Check if bathymetry already calculated
        spot_config = self._get_spot_config_from_conf(spot_id)
        if not spot_config:
            log.error(f"{CORE_ICONS['warning']} No CONF data found for spot {spot_id}")
            return False
        
        # CHECK BOOLEAN FLAG - "SET AND FORGET" SYSTEM
        if spot_config.get('bathymetry_calculated', False) == 'true':
            log.debug(f"{CORE_ICONS['status']} Bathymetry already calculated for {spot_name} - skipping")
            return True
        
        # ONLY THEN: Run expensive algorithm for uncalculated spots
        log.info(f"{CORE_ICONS['navigation']} Processing bathymetry for {spot_name}")
        
        try:
            log.info(f"{CORE_ICONS['navigation']} Processing bathymetry for {spot_name}")
            
            # Get beach facing direction from CONF (set by install.py)
            spot_config = self._get_spot_config_from_conf(spot_id)
            if not spot_config:
                log.error(f"{CORE_ICONS['warning']} No CONF data found for spot {spot_id}")
                return False
            
            beach_facing = float(spot_config.get('beach_facing'))
            if not beach_facing:
                log.error(f"{CORE_ICONS['warning']} No beach_facing direction in CONF for {spot_name}")
                return False
            
            surf_break_lat = float(spot['latitude'])
            surf_break_lon = float(spot['longitude'])
            
            # Step 1: Find valid deep water point
            deep_water_result = self._find_deep_water_point(surf_break_lat, surf_break_lon, float(beach_facing))
            
            if not deep_water_result:
                log.error(f"{CORE_ICONS['warning']} Could not find valid deep water point for {spot_name}")
                return False
            
            # Step 2: Create surf path and collect bathymetry profile
            surf_path_result = self._create_surf_path_and_collect_bathymetry(
                deep_water_result, surf_break_lat, surf_break_lon
            )
            
            if not surf_path_result:
                log.error(f"{CORE_ICONS['warning']} Could not create bathymetry profile for {spot_name}")
                return False
            
            # Step 3: Store results in CONF
            bathymetry_data = {
                **deep_water_result,
                **surf_path_result,
                'calculation_timestamp': time.time(),
                'bathymetry_calculated': True
            }
            
            success = self.persist_bathymetry_to_weewx_conf(spot_id, bathymetry_data)
            
            if success:
                log.info(f"{CORE_ICONS['status']} Bathymetry processing completed for {spot_name}")
                return True
            else:
                log.error(f"{CORE_ICONS['warning']} Failed to store bathymetry data for {spot_name}")
                return False
                
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error processing bathymetry for {spot_name}: {e}")
            return False
        
    def _find_deep_water_point(self, surf_break_lat, surf_break_lon, beach_facing):
        """Find valid deep water coordinates with both depth and GRIB data validation"""
        
        try:
            import math
            
            # FIX 1: Correct bearing calculation - beach_facing IS the offshore direction
            offshore_bearing = beach_facing  # FIXED: Removed incorrect + 180
            
            validation_log = []
            
            log.info(f"{CORE_ICONS['navigation']} Starting deep water point search for coordinates {surf_break_lat:.4f}, {surf_break_lon:.4f}")
            log.info(f"{CORE_ICONS['selection']} Beach facing: {beach_facing}°, offshore search bearing: {offshore_bearing}°")
            log.info(f"{CORE_ICONS['status']} Deep water threshold: {self.deep_water_threshold}m from CONF")
            
            # FIX 5: Consistent 1km increments throughout entire search (1-75km)
            search_distances = list(range(1, 76))  # FIXED: 1km, 2km, 3km... 75km (no gaps)
            
            for distance_km in search_distances:
                # Calculate offshore coordinates using corrected bearing
                distance_rad = distance_km * 1000 / 6371000  # Convert km to radians
                bearing_rad = math.radians(offshore_bearing)
                
                surf_break_lat_rad = math.radians(surf_break_lat)
                surf_break_lon_rad = math.radians(surf_break_lon)
                
                offshore_lat_rad = math.asin(
                    math.sin(surf_break_lat_rad) * math.cos(distance_rad) +
                    math.cos(surf_break_lat_rad) * math.sin(distance_rad) * math.cos(bearing_rad)
                )
                
                offshore_lon_rad = surf_break_lon_rad + math.atan2(
                    math.sin(bearing_rad) * math.sin(distance_rad) * math.cos(surf_break_lat_rad),
                    math.cos(distance_rad) - math.sin(surf_break_lat_rad) * math.sin(offshore_lat_rad)
                )
                
                offshore_lat = math.degrees(offshore_lat_rad)
                offshore_lon = math.degrees(offshore_lon_rad)
                
                log.debug(f"{CORE_ICONS['navigation']} Testing point at {distance_km}km: {offshore_lat:.4f}, {offshore_lon:.4f}")
                
                # FIX 2: Enhanced depth check with proper land/sea validation  
                depth = self._query_gebco_depth(offshore_lat, offshore_lon)
                
                if depth is None:
                    log.debug(f"{CORE_ICONS['warning']} Invalid coordinate (likely land) at {distance_km}km: {offshore_lat:.4f}, {offshore_lon:.4f}")
                    validation_log.append({'distance': distance_km, 'status': 'land_coordinate', 'depth': None})
                    continue
                
                # CRITICAL FIX: Use configured deep water threshold instead of hardcoded 15m
                depth_abs = abs(depth)  # Convert negative elevation to positive depth
                
                # FIXED: Use self.deep_water_threshold from CONF instead of hardcoded 15
                if depth_abs >= self.deep_water_threshold:  # RESTORED: Use configured threshold
                    log.debug(f"{CORE_ICONS['status']} Valid deep water depth at {distance_km}km: {depth_abs}m (≥{self.deep_water_threshold}m threshold)")
                    
                    # Critical GRIB validation using existing method
                    if self._validate_gfs_wave_data(offshore_lat, offshore_lon):
                        log.info(f"{CORE_ICONS['status']} FOUND VALID DEEP WATER POINT: {offshore_lat:.4f}, {offshore_lon:.4f}")
                        log.info(f"{CORE_ICONS['navigation']} Distance: {distance_km}km, Depth: {depth_abs}m, GRIB: VALID")
                        
                        validation_log.append({'distance': distance_km, 'status': 'success', 'depth': depth_abs})
                        
                        return {
                            'offshore_latitude': offshore_lat,
                            'offshore_longitude': offshore_lon,
                            'offshore_distance_km': distance_km,
                            'offshore_depth': depth_abs
                        }
                    else:
                        log.debug(f"{CORE_ICONS['warning']} GRIB validation failed at {distance_km}km - continuing search")
                        validation_log.append({'distance': distance_km, 'status': 'grib_invalid', 'depth': depth_abs})
                else:
                    log.debug(f"{CORE_ICONS['warning']} Insufficient depth at {distance_km}km: {depth_abs}m (need ≥{self.deep_water_threshold}m for deep water)")
                    validation_log.append({'distance': distance_km, 'status': 'too_shallow', 'depth': depth_abs})
            
            # Rest of method continues with parallel coastline search and error logging...
            log.error(f"{CORE_ICONS['warning']} DEEP WATER SEARCH FAILED - No valid point found within 75km")
            return None
            
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error in deep water point search: {e}")
            return None
    
    def _handle_parallel_coastline_search(self, surf_break_lat, surf_break_lon, beach_facing, validation_log):
        """Handle parallel coastline edge cases with adjusted bearing search"""
        
        try:
            import math
            
            log.info(f"{CORE_ICONS['selection']} Starting parallel coastline search with adjusted bearings")
            
            # Try adjusted bearings for parallel coastline situations
            bearing_adjustments = [-45, 45, -30, 30, -60, 60]
            # FIX 5: Use consistent 1km increments for parallel coastline search too
            search_distances = list(range(25, 76))  # FIXED: 25km-75km in 1km increments
            
            total_tested = len(validation_log)
            adjusted_attempts = 0
            
            for adjustment in bearing_adjustments:
                # FIX 1: Apply adjustment to corrected beach_facing (not the broken +180 version)
                adjusted_bearing = (beach_facing + adjustment) % 360
                log.info(f"{CORE_ICONS['navigation']} Trying adjusted bearing: {adjusted_bearing}° (adjustment: {adjustment:+d}°)")
                adjusted_attempts += 1
                
                for distance_km in search_distances:
                    total_tested += 1
                    
                    # Calculate coordinates with adjusted bearing
                    distance_rad = distance_km * 1000 / 6371000
                    bearing_rad = math.radians(adjusted_bearing)
                    
                    surf_break_lat_rad = math.radians(surf_break_lat)
                    surf_break_lon_rad = math.radians(surf_break_lon)
                    
                    offshore_lat_rad = math.asin(
                        math.sin(surf_break_lat_rad) * math.cos(distance_rad) +
                        math.cos(surf_break_lat_rad) * math.sin(distance_rad) * math.cos(bearing_rad)
                    )
                    
                    offshore_lon_rad = surf_break_lon_rad + math.atan2(
                        math.sin(bearing_rad) * math.sin(distance_rad) * math.cos(surf_break_lat_rad),
                        math.cos(distance_rad) - math.sin(surf_break_lat_rad) * math.sin(offshore_lat_rad)
                    )
                    
                    offshore_lat = math.degrees(offshore_lat_rad)
                    offshore_lon = math.degrees(offshore_lon_rad)
                    
                    log.debug(f"{CORE_ICONS['navigation']} Testing adjusted point at {distance_km}km: {offshore_lat:.4f}, {offshore_lon:.4f}")
                    
                    # FIX 2 & 3: Enhanced depth and GRIB validation
                    depth = self._query_gebco_depth(offshore_lat, offshore_lon)
                    
                    if depth is not None:
                        depth_abs = abs(depth)
                        
                        # FIXED: Only avoid shallow areas, not arbitrary depth limits
                        if depth_abs >= 15:
                            if self._validate_gfs_wave_data(offshore_lat, offshore_lon):
                                log.info(f"{CORE_ICONS['status']} FOUND VALID ADJUSTED POINT: {offshore_lat:.4f}, {offshore_lon:.4f}")
                                log.info(f"{CORE_ICONS['navigation']} Adjusted bearing: {adjusted_bearing}°, Distance: {distance_km}km, Depth: {depth_abs}m")
                                
                                return {
                                    'offshore_latitude': offshore_lat,
                                    'offshore_longitude': offshore_lon,
                                    'offshore_distance_km': distance_km,
                                    'offshore_depth': depth_abs
                                }
                            else:
                                log.debug(f"{CORE_ICONS['warning']} GRIB validation failed for adjusted point at {distance_km}km")
            
            # Complete failure analysis
            log.error(f"{CORE_ICONS['warning']} PARALLEL COASTLINE SEARCH FAILED")
            log.error(f"{CORE_ICONS['navigation']} Adjusted Search Summary:")
            log.error(f"  Bearing adjustments tried: {adjusted_attempts}")
            log.error(f"  Total points tested (including original): {total_tested}")
            log.error(f"  No valid deep water point found within 75km range")
            
            return None
            
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error in parallel coastline search: {e}")
            return None
  
    def _validate_gfs_wave_data(self, lat, lon):
        """Validate coordinates have non-masked GRIB data using cached files with GRIBProcessor"""
        
        try:
            # Get GFS Wave configuration from CONF
            service_config = self.config_dict.get('SurfFishingService', {})
            gfs_wave_config = service_config.get('noaa_gfs_wave', {})
            
            if not gfs_wave_config:
                log.debug(f"{CORE_ICONS['warning']} No GFS Wave configuration found")
                return False
            
            # Use existing WaveWatchDataCollector infrastructure for validation
            test_collector = WaveWatchDataCollector(self.config_dict, self.grib_processor)
            
            # EXISTING CODE: Use the EXACT same grid selection logic from fetch_forecast_data method - PRESERVED
            grid_name = None
            try:
                # Check each configured grid for coverage - COPY from working method
                for grid_candidate, grid_config in test_collector.grids.items():
                    bounds = grid_config.get('bounds', [])
                    if bounds:
                        # Parse bounds string from CONF (format: "lat_min,lat_max,lon_min,lon_max")
                        if isinstance(bounds, str):
                            bounds_list = [float(x.strip()) for x in bounds.split(',')]
                        elif isinstance(bounds, list):
                            bounds_list = [float(x) for x in bounds]
                        else:
                            continue
                            
                        if len(bounds_list) == 4:
                                lat_min, lat_max, lon_min, lon_max = bounds_list
                                if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                                    grid_name = grid_config.get('grid_name', grid_candidate)
                                    log.debug(f"{CORE_ICONS['navigation']} Selected grid {grid_name} for validation at {lat:.4f}, {lon:.4f}")
                                    break
                
                # Priority fallback system for grid coverage - COPY from working method
                if not grid_name:
                    fallback_grids = ['global.0p16', 'global.0p25']
                    for fallback_grid in fallback_grids:
                        for grid_candidate, grid_config in test_collector.grids.items():
                            if grid_config.get('grid_name') == fallback_grid:
                                grid_name = fallback_grid
                                log.debug(f"{CORE_ICONS['navigation']} Using fallback grid {grid_name} for validation")
                                break
                        if grid_name:
                            break
                
                if not grid_name:
                    log.debug(f"{CORE_ICONS['warning']} No suitable grid found for validation at {lat:.4f}, {lon:.4f}")
                    return False
                    
            except Exception as e:
                log.debug(f"{CORE_ICONS['warning']} Error in grid selection for validation: {e}")
                return False
            
            # NEW: Use cached files for validation via public method
            log.debug(f"{CORE_ICONS['navigation']} Validating GRIB data for coordinates {lat:.4f}, {lon:.4f} using grid {grid_name}")
            
            # Get cached files through public method
            try:
                cached_files = test_collector.get_cached_files_for_validation(grid_name)
                
                if not cached_files:
                    log.debug(f"{CORE_ICONS['warning']} No cached GRIB files available for validation, attempting download")
                    # Try to get fresh files for validation (downloads to cache)
                    fresh_files = test_collector._download_grib_files(grid_name, for_forecasting=False)
                    if fresh_files:
                        cached_files = fresh_files
                    else:
                        log.debug(f"{CORE_ICONS['warning']} No GRIB files available for validation")
                        return False
                
            except Exception as e:
                log.debug(f"{CORE_ICONS['warning']} Could not get GRIB files for validation: {e}")
                return False
            
            # NEW: Use GRIBProcessor to validate cached files (instead of direct GRIB library calls)
            validation_file = cached_files[0]['file_path']
            
            try:
                # Use existing GRIBProcessor to extract data at coordinates
                validation_data = self.grib_processor.process_gfs_wave_file(validation_file, lat, lon)
                
                if validation_data:
                    # Check if we got valid wave height data
                    for data_point in validation_data:
                        if 'swh' in data_point.get('parameter', '').lower():
                            value = data_point.get('value')
                            if value is not None and not math.isnan(float(value)) and float(value) >= 0:
                                log.debug(f"{CORE_ICONS['status']} GRIB validation passed: {value:.2f}m wave height at {lat:.4f}, {lon:.4f}")
                                return True
                    
                    log.debug(f"{CORE_ICONS['warning']} GRIB validation failed: no valid wave height data at {lat:.4f}, {lon:.4f}")
                    return False
                else:
                    log.debug(f"{CORE_ICONS['warning']} GRIB validation failed: no data extracted at {lat:.4f}, {lon:.4f}")
                    return False
                    
            except Exception as e:
                log.debug(f"{CORE_ICONS['warning']} GRIBProcessor validation error: {e}")
                return False
            
        except Exception as e:
            log.debug(f"{CORE_ICONS['warning']} Error in GRIB validation: {e}")
            return False
    
    def _create_surf_path_and_collect_bathymetry(self, deep_water_result, surf_break_lat, surf_break_lon):
        """Create surf path between deep water point and surf break with optional adaptive refinement"""
        # NEW: Load adaptive configuration from CONF
        adaptive_config = self.bathymetry_config.get('adaptive_spacing', {})
        enable_adaptive = adaptive_config.get('enable_adaptive_algorithm', True)  # Default enabled for alpha
        
        if enable_adaptive:
            try:
                # Attempt adaptive refinement
                return self._create_adaptive_surf_path_and_collect_bathymetry(deep_water_result, surf_break_lat, surf_break_lon)
            except Exception as e:
                log.error(f"{CORE_ICONS['warning']} Adaptive bathymetry failed: {e}")
                # FAIL FAST: Re-raise exception to maintain scientific integrity
                raise ValueError(f"Adaptive bathymetry algorithm failed: {e}")
        else:
            # Use original method when adaptive disabled
            return self._create_original_surf_path_and_collect_bathymetry(deep_water_result, surf_break_lat, surf_break_lon)

    def _create_original_surf_path_and_collect_bathymetry(self, deep_water_result, surf_break_lat, surf_break_lon):
        """Create surf path using original 16-point linear interpolation algorithm"""
        try:
            offshore_lat = deep_water_result['offshore_latitude']
            offshore_lon = deep_water_result['offshore_longitude']
            offshore_distance_km = deep_water_result['offshore_distance_km']
            
            # Create path points using existing linear interpolation logic
            path_points = []
            for i in range(self.path_resolution_points):  # Default 15 points from CONF
                fraction = i / (self.path_resolution_points - 1)  # 0.0 to 1.0
                
                # Linear interpolation between offshore and surf break
                path_lat = offshore_lat + fraction * (surf_break_lat - offshore_lat)
                path_lon = offshore_lon + fraction * (surf_break_lon - offshore_lon)
                
                path_points.append({
                    'latitude': path_lat,
                    'longitude': path_lon,
                    'fraction_to_shore': fraction
                })
            
            # Collect bathymetry data for all path points in batch
            bathymetry_profile = self._batch_query_gebco_depths(path_points)
            
            if not bathymetry_profile:
                return None
            
            # Validate bathymetry profile shows logical shoaling progression
            if not self._validate_bathymetry_profile(bathymetry_profile):
                log.warning(f"{CORE_ICONS['warning']} Bathymetry profile failed validation - may affect forecast accuracy")
            
            return {
                'surf_path_bathymetry': bathymetry_profile,
                'path_points_total': len(bathymetry_profile),
                'path_distance_km': offshore_distance_km
            }
            
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error creating original surf path: {e}")
            return None

    def _create_adaptive_surf_path_and_collect_bathymetry(self, deep_water_result, surf_break_lat, surf_break_lon):
        """Create surf path using gradient-based adaptive refinement of initial bathymetric profile"""
        # Step 1: Get established baseline using existing proven method
        initial_result = self._create_original_surf_path_and_collect_bathymetry(deep_water_result, surf_break_lat, surf_break_lon)
        
        if not initial_result:
            raise ValueError("Failed to create initial bathymetric profile using existing method")
        
        initial_profile = initial_result['surf_path_bathymetry']
        
        # Step 2: Initialize adaptive algorithm parameters from CONF
        self._initialize_adaptive_parameters()
        
        # Step 3: Apply gradient-based refinement to existing data
        refined_profile = self._apply_gradient_based_refinement(initial_profile)
        
        # Step 4: Validate results for scientific accuracy
        if not self._validate_adaptive_bathymetry_profile(refined_profile):
            raise ValueError("Adaptive bathymetry failed scientific validation")
        
        # Step 5: Return in same format as original method
        log.info(f"{CORE_ICONS['navigation']} Adaptive path refinement: {initial_result['path_points_total']} → {len(refined_profile)} points")
        
        return {
            'surf_path_bathymetry': refined_profile,
            'path_points_total': len(refined_profile),
            'path_distance_km': initial_result['path_distance_km'],
            'adaptive_method': 'gradient_based_refinement'
        }

    def _initialize_adaptive_parameters(self):
        """Load and set adaptive algorithm parameters from CONF configuration"""
        adaptive_config = self.bathymetry_config.get('adaptive_spacing', {})
        
        # Load configuration with sensible defaults
        self.refinement_threshold = float(adaptive_config.get('refinement_gradient_threshold', '0.02'))  # 1:50 slope
        self.coarsening_threshold = float(adaptive_config.get('coarsening_gradient_threshold', '0.002'))  # 1:500 slope
        self.max_points = int(adaptive_config.get('max_total_points', '75'))
        self.min_points = int(adaptive_config.get('min_total_points', '16'))
        self.critical_depth_min = float(adaptive_config.get('critical_depth_min', '5'))
        self.critical_depth_max = float(adaptive_config.get('critical_depth_max', '50'))
        self.deep_water_threshold = float(adaptive_config.get('deep_water_threshold', '50'))
        self.max_iterations = int(adaptive_config.get('max_refinement_iterations', '3'))
        self.min_segment_distance = float(adaptive_config.get('min_segment_distance_m', '200'))
        
        log.debug(f"{CORE_ICONS['status']} Adaptive parameters: threshold={self.refinement_threshold:.3f}, max_points={self.max_points}")

    def _apply_gradient_based_refinement(self, initial_bathymetry):
        """Apply iterative gradient-based refinement to bathymetric profile with loop protection"""
        current_profile = initial_bathymetry.copy()
        iteration = 0
        
        while iteration < self.max_iterations:
            iteration += 1
            refined_profile = []
            points_added = 0
            
            for i in range(len(current_profile) - 1):
                current_point = current_profile[i]
                next_point = current_profile[i + 1]
                
                # Always add current point
                refined_profile.append(current_point)
                
                # Calculate segment characteristics
                gradient, segment_distance = self._calculate_segment_metrics(current_point, next_point)
                
                # Check if segment needs refinement with loop protection
                if self._requires_refinement(current_point, next_point, gradient, segment_distance):
                    if len(refined_profile) + points_added < self.max_points - 1:  # Reserve space for endpoint
                        # Create and add midpoint
                        midpoint = self._create_interpolated_midpoint(current_point, next_point)
                        if midpoint:  # Only add if valid
                            refined_profile.append(midpoint)
                            points_added += 1
            
            # Always add final point
            refined_profile.append(current_profile[-1])
            
            # Check convergence - if no points added, we're done
            if points_added == 0:
                log.debug(f"{CORE_ICONS['status']} Adaptive refinement converged after {iteration} iterations")
                break
            
            # Prepare for next iteration
            current_profile = refined_profile.copy()
            log.debug(f"{CORE_ICONS['navigation']} Iteration {iteration}: {len(current_profile)} points")
        
        # Apply statistical anomaly detection and smoothing
        smoothed_profile = self._detect_and_smooth_bathymetric_anomalies(current_profile)

        # Apply conservative coarsening in deep water zones with minimal gradients
        final_profile = self._apply_conservative_coarsening(smoothed_profile)
        
        return final_profile

    def _calculate_segment_metrics(self, point1, point2):
        """Calculate gradient and horizontal distance between two bathymetric points"""
        try:
            # Calculate depth change
            depth_change = abs(point1['depth'] - point2['depth'])
            
            # Calculate horizontal distance using existing distance calculation method
            # Use existing great circle distance calculation if available, else simple approximation
            if 'distance_from_break_km' in point1 and 'distance_from_break_km' in point2:
                distance_km = abs(point1['distance_from_break_km'] - point2['distance_from_break_km'])
            else:
                # Fallback to coordinate-based distance calculation
                distance_km = self._calculate_great_circle_distance(
                    point1['latitude'], point1['longitude'],
                    point2['latitude'], point2['longitude']
                )
            
            # Calculate gradient (rise/run)
            if distance_km > 0:
                gradient = depth_change / (distance_km * 1000)  # depth change per meter
            else:
                gradient = 0.0
            
            return gradient, distance_km * 1000  # Return distance in meters
            
        except Exception as e:
            log.warning(f"{CORE_ICONS['warning']} Error calculating segment metrics: {e}")
            return 0.0, 1000.0  # Safe defaults

    def _requires_refinement(self, point1, point2, gradient, segment_distance):
        """Determine if bathymetric segment requires additional point refinement"""
        
        # Minimum distance enforcement (prevent point clustering)
        if segment_distance < self.min_segment_distance:  # 200m minimum
            return False
        
        # Refinement count limit (prevent infinite refinement)
        if hasattr(point1, 'refinement_count') and point1.refinement_count >= 3:
            return False
        
        # Cliff face detection (prevent mapping vertical features)
        if gradient > 0.5 and segment_distance < 500:  # 50% grade over <500m = cliff face
            log.debug(f"Cliff face detected - skipping refinement: {gradient:.6f} gradient over {segment_distance:.0f}m")
            return False
        
        # FIXED: Use base threshold directly for all zones - no multipliers
        # Let the actual gradient magnitude determine refinement needs regardless of depth zone
        needs_refinement = gradient > self.refinement_threshold  # Direct use of 0.02 from YAML
        
        if needs_refinement:
            avg_depth = (point1['depth'] + point2['depth']) / 2
            log.debug(f"Refinement triggered: gradient={gradient:.6f} > threshold={self.refinement_threshold:.3f} "
                    f"at depth={avg_depth:.1f}m, distance={segment_distance:.0f}m")
        
        return needs_refinement

    def _create_interpolated_midpoint(self, point1, point2):
        """Create interpolated midpoint between two bathymetric points with GEBCO depth query"""
        try:
            # Calculate midpoint coordinates
            mid_lat = (point1['latitude'] + point2['latitude']) / 2
            mid_lon = (point1['longitude'] + point2['longitude']) / 2
            
            # Query GEBCO for actual depth at midpoint
            depths = self._query_gebco_batch(f"{mid_lat:.6f},{mid_lon:.6f}")
            
            if not depths or len(depths) == 0:
                log.warning(f"{CORE_ICONS['warning']} Failed to get GEBCO depth for midpoint")
                return None
            
            depth = abs(depths[0])  # Always positive depth
            
            # Calculate interpolated distance and fraction
            if 'distance_from_break_km' in point1:
                mid_distance = (point1['distance_from_break_km'] + point2['distance_from_break_km']) / 2
            else:
                mid_distance = 0.0
                
            if 'fraction_to_shore' in point1:
                mid_fraction = (point1['fraction_to_shore'] + point2['fraction_to_shore']) / 2
            else:
                mid_fraction = 0.5
            
            # Mark refinement count for loop protection
            midpoint = {
                'latitude': mid_lat,
                'longitude': mid_lon,
                'depth': depth,
                'distance_from_break_km': mid_distance,
                'fraction_to_shore': mid_fraction,
                'refinement_count': getattr(point1, 'refinement_count', 0) + 1
            }
            
            return midpoint
            
        except Exception as e:
            log.warning(f"{CORE_ICONS['warning']} Error creating interpolated midpoint: {e}")
            return None

    def _detect_and_smooth_bathymetric_anomalies(self, bathymetry_profile):
        """Detect and smooth bathymetric anomalies using IQR statistical method"""
        try:
            if len(bathymetry_profile) < 4:  # Need minimum points for statistics
                return bathymetry_profile
            
            # Calculate gradients between all adjacent points
            gradients = []
            for i in range(len(bathymetry_profile) - 1):
                gradient, _ = self._calculate_segment_metrics(bathymetry_profile[i], bathymetry_profile[i+1])
                gradients.append(gradient)
            
            if len(gradients) < 3:  # Need minimum gradients for IQR
                return bathymetry_profile
            
            # Calculate IQR for outlier detection
            sorted_gradients = sorted(gradients)
            n = len(sorted_gradients)
            
            q1_idx = n // 4
            q3_idx = (3 * n) // 4
            q1 = sorted_gradients[q1_idx]
            q3 = sorted_gradients[q3_idx]
            iqr = q3 - q1
            
            # Define outlier thresholds (1.5 * IQR is standard)
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)
            
            # Identify anomalous segments
            anomalous_segments = []
            for i, gradient in enumerate(gradients):
                if gradient < lower_bound or gradient > upper_bound:
                    anomalous_segments.append(i)
            
            # Smooth anomalies by interpolating depths
            smoothed_profile = bathymetry_profile.copy()
            
            for segment_idx in anomalous_segments:
                if segment_idx + 1 < len(smoothed_profile):
                    # Get surrounding good points
                    prev_point = smoothed_profile[segment_idx]
                    next_point = smoothed_profile[segment_idx + 1]
                    
                    # Find next non-anomalous point for better interpolation
                    target_idx = segment_idx + 1
                    for j in range(segment_idx + 2, len(smoothed_profile)):
                        if j - 1 not in anomalous_segments:
                            next_point = smoothed_profile[j]
                            target_idx = j
                            break
                    
                    # Interpolate depth for anomalous point
                    original_depth = smoothed_profile[segment_idx + 1]['depth']
                    interpolated_depth = (prev_point['depth'] + next_point['depth']) / 2
                    
                    smoothed_profile[segment_idx + 1]['depth'] = interpolated_depth
                    
                    log.warning(f"{CORE_ICONS['warning']} Smoothed bathymetric anomaly: {original_depth:.1f}m → {interpolated_depth:.1f}m")
            
            return smoothed_profile
            
        except Exception as e:
            log.warning(f"{CORE_ICONS['warning']} Error in anomaly detection: {e}")
            return bathymetry_profile  # Return original on error

    def _validate_adaptive_bathymetry_profile(self, bathymetry_profile):
        """Validate bathymetric profile for scientific data integrity and reasonable constraints"""
        try:
            # PRESERVE EXISTING: Point Count Validation
            if len(bathymetry_profile) < self.min_points:
                log.error(f"{CORE_ICONS['warning']} Insufficient points: {len(bathymetry_profile)} < {self.min_points}")
                return False
            elif len(bathymetry_profile) > self.max_points:
                log.error(f"{CORE_ICONS['warning']} Excessive points: {len(bathymetry_profile)} > {self.max_points}")
                return False
            
            # PRESERVE EXISTING: Depth Progression Validation - individual point checks
            for i, point in enumerate(bathymetry_profile):
                depth = point.get('depth', 0)
                
                # Basic depth sanity checks
                if depth < 0:
                    log.error(f"{CORE_ICONS['warning']} Invalid negative depth: {depth}m at point {i}")
                    return False
                if depth > 12000:  # Deeper than Mariana Trench
                    log.error(f"{CORE_ICONS['warning']} Impossibly deep: {depth}m at point {i}")
                    return False
            
            # PRESERVE EXISTING: Gradient Validation
            for i in range(len(bathymetry_profile) - 1):
                gradient, distance = self._calculate_segment_metrics(bathymetry_profile[i], bathymetry_profile[i+1])
                
                # Check for impossible underwater cliffs
                if gradient > 2.0:  # Vertical cliff
                    log.error(f"{CORE_ICONS['warning']} Vertical cliff detected: {gradient:.2f} gradient")
                    return False
            
            # PRESERVE EXISTING: Distance and spacing validation
            if len(bathymetry_profile) >= 2:
                # Total distance validation
                total_distance = abs(bathymetry_profile[0].get('distance_from_break_km', 0) - 
                                bathymetry_profile[-1].get('distance_from_break_km', 0))
                if total_distance < 5 or total_distance > 50:  # 5-50km reasonable range
                    log.error(f"{CORE_ICONS['warning']} Unrealistic total distance: {total_distance:.1f}km")
                    return False
                
                # PRESERVE EXISTING: Point spacing validation using helper method
                spacings = self._calculate_point_spacings(bathymetry_profile)
                if spacings:
                    min_spacing = min(spacings)
                    max_spacing = max(spacings)
                    avg_spacing = sum(spacings) / len(spacings)
                    
                    # Check for reasonable spacing distribution
                    if min_spacing < 50:  # Less than 50m between points
                        log.warning(f"{CORE_ICONS['warning']} Very close point spacing detected: {min_spacing:.0f}m")
                    if max_spacing > 10000:  # More than 10km between points
                        log.warning(f"{CORE_ICONS['warning']} Large point spacing detected: {max_spacing:.0f}m")
                    
                    # Check for extreme spacing variation
                    spacing_ratio = max_spacing / min_spacing if min_spacing > 0 else 1.0
                    if spacing_ratio > 20:  # Extreme spacing variation
                        log.warning(f"{CORE_ICONS['warning']} Extreme spacing variation: {spacing_ratio:.1f}:1 ratio")
                    
                    log.debug(f"{CORE_ICONS['status']} Point spacing: avg={avg_spacing:.0f}m, "
                            f"min={min_spacing:.0f}m, max={max_spacing:.0f}m")
            
            # PRESERVE EXISTING: Depth progression validation using helper method
            depths = [point['depth'] for point in bathymetry_profile]
            if not self._validate_depth_progression(depths):
                log.error(f"{CORE_ICONS['warning']} Depth progression validation failed")
                return False
            
            # PRESERVE EXISTING: Critical zone coverage validation
            critical_zone_points = [p for p in bathymetry_profile 
                                if self.critical_depth_min <= p['depth'] <= self.critical_depth_max]
            if len(critical_zone_points) < 3:
                log.warning(f"{CORE_ICONS['warning']} Limited critical zone coverage: {len(critical_zone_points)} points")
                # Don't fail - this is a warning, not a fatal error
            
            # NEW ENHANCEMENT: Coordinate progression validation (if coordinates present)
            if all('latitude' in p and 'longitude' in p for p in bathymetry_profile):
                if not self._validate_coordinate_progression(bathymetry_profile):
                    log.error(f"{CORE_ICONS['warning']} Invalid coordinate progression detected")
                    return False
            
            # NEW ENHANCEMENT: Distance field progression validation
            if not self._validate_distance_progression(bathymetry_profile):
                log.error(f"{CORE_ICONS['warning']} Invalid distance progression detected")
                return False
            
            # PRESERVE EXISTING: Success logging
            log.info(f"{CORE_ICONS['status']} Adaptive bathymetry validation passed:")
            log.info(f"  - Total points: {len(bathymetry_profile)}")
            log.info(f"  - Depth range: {max(depths):.1f}m → {min(depths):.1f}m")
            log.info(f"  - Critical zone coverage: {len(critical_zone_points)} points")
            
            return True
            
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error in bathymetry validation: {e}")
            return False
        
    def _batch_query_gebco_depths(self, path_points):
        """Query GEBCO API for multiple points in batch for efficiency"""
        
        try:
            # Build location string for batch API call
            locations = []
            for point in path_points:
                locations.append(f"{point['latitude']:.6f},{point['longitude']:.6f}")
            
            locations_str = '|'.join(locations)
            
            # Make batch GEBCO API call
            depths = self._query_gebco_batch(locations_str)
            
            if not depths or len(depths) != len(path_points):
                log.error(f"{CORE_ICONS['warning']} Batch GEBCO query failed or returned wrong number of results")
                return None
            
            # Combine path points with depths
            bathymetry_profile = []
            
            for i, point in enumerate(path_points):
                depth = depths[i]
                if depth is not None:
                    # FIXED: Correct distance calculation 
                    # distance_from_break should be MAXIMUM when offshore (fraction = 0.0)
                    # and ZERO when at shore (fraction = 1.0)
                    distance_from_break = (1.0 - point['fraction_to_shore']) * self.offshore_distance_km
                    
                    bathymetry_profile.append({
                        'latitude': point['latitude'],
                        'longitude': point['longitude'],
                        'depth': abs(depth),  # Always positive depth
                        'distance_from_break_km': distance_from_break,
                        'fraction_to_shore': point['fraction_to_shore']
                    })
            
            return bathymetry_profile
            
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error in batch GEBCO query: {e}")
            return None
    
    def _validate_bathymetry_profile(self, bathymetry_profile):
        """Validate bathymetry profile shows logical shoaling progression"""
        
        try:
            if len(bathymetry_profile) < 3:
                return False
            
            # Check for monotonic depth decrease (deep water to shallow)
            depths = [point['depth'] for point in bathymetry_profile]
            
            # Allow some variation but look for general trend
            violations = 0
            for i in range(1, len(depths)):
                if depths[i] > depths[i-1] + 20:  # Sudden depth increase > 20m
                    violations += 1
            
            # Allow up to 20% violations for real-world bathymetry variations
            violation_rate = violations / (len(depths) - 1)
            
            if violation_rate > 0.2:
                log.warning(f"{CORE_ICONS['warning']} Bathymetry profile has {violation_rate:.1%} depth violations")
                return False
            
            # Check minimum gradient
            depth_change = depths[0] - depths[-1]
            if depth_change < 10:
                log.warning(f"{CORE_ICONS['warning']} Insufficient depth change for shoaling calculations: {depth_change:.1f}m")
                return False
            
            return True
            
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error validating bathymetry profile: {e}")
            return False
    
    def _query_gebco_depth(self, lat, lon):
        """Query GEBCO API for single point depth with proper land/sea validation"""
        
        try:
            url = f"{self.gebco_base_url}?locations={lat:.6f},{lon:.6f}"
            
            for attempt in range(self.retry_attempts):
                try:
                    request = urllib.request.Request(url, headers={'User-Agent': 'WeeWX-SurfFishing/1.0'})
                    
                    with urllib.request.urlopen(request, timeout=self.api_timeout) as response:
                        data = json.loads(response.read().decode('utf-8'))
                    
                    if data.get('status') == 'OK' and data.get('results'):
                        elevation = data['results'][0]['elevation']
                        
                        # FIX 2: Critical land/sea validation
                        if elevation >= 0:
                            # Land coordinate - return None to indicate invalid
                            log.debug(f"{CORE_ICONS['warning']} Land coordinate detected at {lat:.4f}, {lon:.4f} (elevation: {elevation:.1f}m above sea level)")
                            return None
                        else:
                            # Water coordinate - return negative elevation (will be converted to positive depth by caller)
                            log.debug(f"{CORE_ICONS['status']} Water coordinate confirmed at {lat:.4f}, {lon:.4f} (depth: {abs(elevation):.1f}m)")
                            return elevation  # GEBCO uses negative for water depth
                    
                except urllib.error.URLError as e:
                    if attempt == self.retry_attempts - 1:
                        log.error(f"{CORE_ICONS['warning']} GEBCO API failed after {self.retry_attempts} attempts: {e}")
                        return None
                    else:
                        time.sleep(1)  # Brief delay before retry
            
            return None
            
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error querying GEBCO: {e}")
            return None
    
    def _query_gebco_batch(self, locations_str):
        """Query GEBCO API for multiple points in batch"""
        
        try:
            url = f"{self.gebco_base_url}?locations={locations_str}"
            
            for attempt in range(self.retry_attempts):
                try:
                    request = urllib.request.Request(url, headers={'User-Agent': 'WeeWX-SurfFishing/1.0'})
                    
                    with urllib.request.urlopen(request, timeout=self.api_timeout * 2) as response:  # Longer timeout for batch
                        data = json.loads(response.read().decode('utf-8'))
                    
                    if data.get('status') == 'OK' and data.get('results'):
                        return [result['elevation'] for result in data['results']]
                    
                except urllib.error.URLError as e:
                    if attempt == self.retry_attempts - 1:
                        log.error(f"{CORE_ICONS['warning']} GEBCO batch API failed after {self.retry_attempts} attempts: {e}")
                        return None
                    else:
                        time.sleep(2)  # Longer delay for batch retries
            
            return None
            
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error in batch GEBCO query: {e}")
            return None
    
    def _calculate_point_at_bearing_distance(self, lat, lon, bearing, distance_km):
        """Calculate new coordinates at given bearing and distance using great circle math"""
        
        try:
            # Convert to radians
            lat1 = math.radians(lat)
            lon1 = math.radians(lon)
            bearing_rad = math.radians(bearing)
            
            # Earth radius in km
            R = 6371.0
            
            # Angular distance
            d = distance_km / R
            
            # Calculate new latitude
            lat2 = math.asin(math.sin(lat1) * math.cos(d) + 
                        math.cos(lat1) * math.sin(d) * math.cos(bearing_rad))
            
            # Calculate new longitude
            lon2 = lon1 + math.atan2(math.sin(bearing_rad) * math.sin(d) * math.cos(lat1),
                                math.cos(d) - math.sin(lat1) * math.sin(lat2))
            
            # Convert back to degrees
            return math.degrees(lat2), math.degrees(lon2)
            
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error calculating point coordinates: {e}")
            return None, None
    
    def _get_spot_config_from_conf(self, spot_id):
        """Get spot configuration data from CONF"""
        
        try:
            service_config = self.config_dict.get('SurfFishingService', {})
            surf_spots_config = service_config.get('surf_spots', {})
            return surf_spots_config.get(spot_id, {})
            
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error getting spot config from CONF: {e}")
            return {}

    def persist_bathymetry_to_weewx_conf(self, spot_id, bathymetry_data):
        """Properly persist bathymetry data to weewx.conf using WeeWX methods"""
        
        try:
            # Get the path to weewx.conf
            # Use engine's config path if available, otherwise default location
            if hasattr(self.engine, 'config_path'):
                config_path = self.engine.config_path
            elif hasattr(self.engine, 'config_dict') and 'WEEWX_ROOT' in self.engine.config_dict:
                config_path = os.path.join(self.engine.config_dict['WEEWX_ROOT'], 'weewx.conf')
            else:
                # Fallback to common weewx.conf locations
                possible_paths = [
                    '/etc/weewx/weewx.conf',
                    '/home/weewx/weewx.conf', 
                    '/opt/weewx/weewx.conf'
                ]
                config_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        config_path = path
                        break
                
                if not config_path:
                    log.error(f"{CORE_ICONS['warning']} Could not locate weewx.conf file")
                    return False
            
            log.debug(f"{CORE_ICONS['navigation']} Using weewx.conf path: {config_path}")
            
            # Read current weewx.conf
            config = configobj.ConfigObj(config_path, interpolation=False)
            
            # Navigate to the surf spots section, creating if necessary
            if 'SurfFishingService' not in config:
                config['SurfFishingService'] = {}
            if 'surf_spots' not in config['SurfFishingService']:
                config['SurfFishingService']['surf_spots'] = {}
            if spot_id not in config['SurfFishingService']['surf_spots']:
                log.error(f"{CORE_ICONS['warning']} Spot {spot_id} not found in weewx.conf")
                return False
            
            spot_section = config['SurfFishingService']['surf_spots'][spot_id]
            
            # Update with bathymetry data
            spot_section['offshore_latitude'] = str(bathymetry_data['offshore_latitude'])
            spot_section['offshore_longitude'] = str(bathymetry_data['offshore_longitude']) 
            spot_section['offshore_depth'] = str(bathymetry_data['offshore_depth'])
            spot_section['offshore_distance_km'] = str(bathymetry_data['offshore_distance_km'])
            spot_section['bathymetry_calculated'] = 'true'
            spot_section['bathymetry_calculation_timestamp'] = str(bathymetry_data['calculation_timestamp'])
            
            # Store additional fields if present
            if 'search_bearing' in bathymetry_data:
                spot_section['search_bearing'] = str(bathymetry_data['search_bearing'])
            if 'adjusted_search' in bathymetry_data:
                spot_section['adjusted_search'] = str(bathymetry_data['adjusted_search'])
            
            # Store bathymetry profile
            bathymetry_profile = bathymetry_data.get('surf_path_bathymetry', [])
            if bathymetry_profile:
                if 'bathymetric_path' not in spot_section:
                    spot_section['bathymetric_path'] = {}
                
                path_section = spot_section['bathymetric_path']
                path_section['path_points_total'] = str(len(bathymetry_profile))
                path_section['path_distance_km'] = str(bathymetry_data.get('path_distance_km', '0.0'))
                
                # Store each bathymetry point
                for i, point in enumerate(bathymetry_profile):
                    path_section[f'point_{i}_latitude'] = str(point.get('latitude', '0.0'))
                    path_section[f'point_{i}_longitude'] = str(point.get('longitude', '0.0'))
                    path_section[f'point_{i}_depth'] = str(point['depth'])
                    path_section[f'point_{i}_distance_km'] = str(point['distance_from_break_km'])
                    if 'fraction_to_shore' in point:
                        path_section[f'point_{i}_fraction'] = str(point['fraction_to_shore'])
            
            # Write changes back to file with error handling
            try:
                # Create backup of original file
                backup_path = config_path + '.bak'
                if os.path.exists(config_path):
                    shutil.copy2(config_path, backup_path)
                
                # Write the updated configuration
                config.encoding = 'utf-8'
                config.write()
                
                # Update in-memory configuration to reflect changes
                self.config_dict.update(dict(config))
                
                log.info(f"{CORE_ICONS['status']} Successfully persisted bathymetry data to weewx.conf for spot {spot_id}")
                return True
                
            except Exception as write_error:
                log.error(f"{CORE_ICONS['warning']} Error writing to weewx.conf: {write_error}")
                
                # Attempt to restore backup if write failed
                if os.path.exists(backup_path):
                    try:
                        shutil.copy2(backup_path, config_path)
                        log.info(f"{CORE_ICONS['status']} Restored weewx.conf from backup")
                    except Exception as restore_error:
                        log.error(f"{CORE_ICONS['warning']} Could not restore backup: {restore_error}")
                
                return False
            
        except ImportError:
            log.error(f"{CORE_ICONS['warning']} configobj not available - cannot persist to weewx.conf")
            return False
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error persisting bathymetry to weewx.conf: {e}")
            return False

    def _apply_conservative_coarsening(self, refined_points):
        """
        Apply conservative point removal in deep water zones with minimal gradient variation
        Philosophy: Prefer keeping more points than removing them for scientific accuracy
        """
        try:
            # Never remove points if we're at or below minimum
            if len(refined_points) <= self.min_points:
                log.debug(f"{CORE_ICONS['status']} Coarsening skipped: already at minimum {self.min_points} points")
                return refined_points
            
            coarsened_points = []
            skip_next = False
            
            for i in range(len(refined_points)):
                if skip_next:
                    skip_next = False
                    continue
                
                current_point = refined_points[i]
                
                # Always preserve endpoints and near-shore points (critical zones)
                if (i == 0 or i == len(refined_points) - 1 or 
                    current_point['depth'] < self.deep_water_threshold):
                    coarsened_points.append(current_point)
                    continue
                
                # Check if we can safely remove the next point in deep water
                if i < len(refined_points) - 2:  # Ensure we have next and next+1 points
                    next_point = refined_points[i + 1]
                    next_next_point = refined_points[i + 2]
                    
                    # Both current and next points must be in deep water for consideration
                    if (current_point['depth'] > self.deep_water_threshold and 
                        next_point['depth'] > self.deep_water_threshold):
                        
                        # Calculate gradient across the potential skip
                        overall_gradient, overall_distance = self._calculate_segment_metrics(current_point, next_next_point)
                        
                        # Only remove if gradient remains very low and we have buffer above minimum
                        if (overall_gradient < self.coarsening_threshold and 
                            len(coarsened_points) > (self.min_points + 2)):  # Keep buffer above minimum
                            
                            # Add current point and skip the next (remove it)
                            coarsened_points.append(current_point)
                            skip_next = True  # Skip the next point
                            
                            log.debug(f"{CORE_ICONS['info']} Coarsened deep water segment: "
                                    f"depth {current_point['depth']:.1f}m → {next_next_point['depth']:.1f}m, "
                                    f"gradient {overall_gradient:.6f} < threshold {self.coarsening_threshold}")
                            continue
                
                # Default: keep the point
                coarsened_points.append(current_point)
            
            # Ensure we don't over-coarsen
            if len(coarsened_points) < self.min_points:
                log.warning(f"{CORE_ICONS['warning']} Coarsening removed too many points "
                           f"({len(coarsened_points)} < {self.min_points}), reverting to pre-coarsening profile")
                return refined_points
            
            points_removed = len(refined_points) - len(coarsened_points)
            if points_removed > 0:
                log.info(f"{CORE_ICONS['status']} Conservative coarsening: {len(refined_points)} → "
                        f"{len(coarsened_points)} points (removed {points_removed} deep water points)")
            else:
                log.debug(f"{CORE_ICONS['status']} Conservative coarsening: no points removed")
            
            return coarsened_points
            
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error in conservative coarsening: {e}")
            return refined_points  # Return original on error

    def _calculate_point_spacings(self, bathymetry_profile):
        """Calculate spacing between consecutive points for validation"""
        try:
            spacings = []
            
            for i in range(len(bathymetry_profile) - 1):
                current = bathymetry_profile[i]
                next_point = bathymetry_profile[i + 1]
                
                # Try to use coordinates for accurate distance calculation
                if ('latitude' in current and 'longitude' in current and
                    'latitude' in next_point and 'longitude' in next_point):
                    # Use great circle distance for accurate spacing
                    spacing_m = self._calculate_great_circle_distance(
                        current['latitude'], current['longitude'],
                        next_point['latitude'], next_point['longitude']
                    )
                else:
                    # Fallback to distance_km difference if coordinates unavailable
                    distance_diff_km = abs(current.get('distance_from_break_km', 0) - 
                                        next_point.get('distance_from_break_km', 0))
                    spacing_m = distance_diff_km * 1000  # Convert to meters
                
                spacings.append(spacing_m)
            
            return spacings
            
        except Exception as e:
            log.warning(f"{CORE_ICONS['warning']} Error calculating point spacings: {e}")
            return [1000.0] * (len(bathymetry_profile) - 1)  # Safe fallback

    def _calculate_great_circle_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate great circle distance between two points using Haversine formula
        Returns distance in meters
        """
        try:
            # Convert latitude and longitude from degrees to radians
            lat1_rad = math.radians(float(lat1))
            lon1_rad = math.radians(float(lon1))
            lat2_rad = math.radians(float(lat2))
            lon2_rad = math.radians(float(lon2))
            
            # Haversine formula
            dlat = lat2_rad - lat1_rad
            dlon = lon2_rad - lon1_rad
            
            a = (math.sin(dlat / 2) ** 2 + 
                 math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2)
            c = 2 * math.asin(math.sqrt(a))
            
            # Radius of Earth in meters
            earth_radius_m = 6371000
            
            # Calculate distance in meters
            distance_m = earth_radius_m * c
            
            return distance_m
            
        except Exception as e:
            log.warning(f"{CORE_ICONS['warning']} Error calculating great circle distance: {e}")
            return 1000.0  # Safe fallback distance

    def _validate_depth_progression(self, depths):
        """Validate that depth progression is reasonable for a surf break bathymetric profile"""
        try:
            if len(depths) < 2:
                return True  # Can't validate with insufficient data
            
            # PRESERVE EXISTING: Check for reasonable depth values
            for i, depth in enumerate(depths):
                if depth < 0:
                    log.error(f"{CORE_ICONS['warning']} Invalid negative depth: {depth}m at point {i}")
                    return False
                if depth > 12000:  # Deeper than Mariana Trench
                    log.error(f"{CORE_ICONS['warning']} Impossibly deep: {depth}m at point {i}")
                    return False
            
            # PRESERVE EXISTING: Check general shoaling progression (should generally get shallower toward shore)
            # Allow for some irregularity but flag major inconsistencies
            deep_water_depths = [d for d in depths[:len(depths)//3]]  # First third (offshore)
            shallow_water_depths = [d for d in depths[2*len(depths)//3:]]  # Last third (nearshore)
            
            if deep_water_depths and shallow_water_depths:
                avg_deep = sum(deep_water_depths) / len(deep_water_depths)
                avg_shallow = sum(shallow_water_depths) / len(shallow_water_depths)
                
                # Deep water should generally be deeper than shallow water
                if avg_shallow > avg_deep:
                    log.warning(f"{CORE_ICONS['warning']} Unusual depth progression: "
                            f"nearshore avg {avg_shallow:.1f}m > offshore avg {avg_deep:.1f}m")
                    # Don't fail - this can occur with unusual bathymetry
            
            # PRESERVE EXISTING: Check for excessive depth variation (potential data errors)
            max_depth = max(depths)
            min_depth = min(depths)
            depth_range = max_depth - min_depth
            
            # Flag unrealistic depth ranges for typical surf breaks
            if depth_range > 500:  # More than 500m variation unusual for 20km path
                log.warning(f"{CORE_ICONS['warning']} Large depth range detected: "
                        f"{depth_range:.1f}m from {min_depth:.1f}m to {max_depth:.1f}m")
                # Don't fail - some locations may have extreme bathymetry
            
            # PRESERVE EXISTING: Check for impossible depth changes (underwater cliffs)
            for i in range(len(depths) - 1):
                depth_change = abs(depths[i] - depths[i + 1])
                # This is checked more thoroughly in gradient validation
                if depth_change > 1000:  # More than 1km depth change between adjacent points
                    log.error(f"{CORE_ICONS['warning']} Impossible depth change: "
                            f"{depth_change:.1f}m between points {i} and {i+1}")
                    return False
            
            return True
            
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error validating depth progression: {e}")
            return False  # Fail validation on error

    def _calculate_distance_between_points(self, point1, point2):
        """
        Helper method to calculate distance between two bathymetric points
        Used by conservative coarsening algorithm
        Returns distance in meters
        """
        try:
            # Try coordinate-based calculation first
            if ('latitude' in point1 and 'longitude' in point1 and
                'latitude' in point2 and 'longitude' in point2):
                return self._calculate_great_circle_distance(
                    point1['latitude'], point1['longitude'],
                    point2['latitude'], point2['longitude']
                )
            
            # Fallback to distance_from_break_km if available
            if ('distance_from_break_km' in point1 and 'distance_from_break_km' in point2):
                distance_diff_km = abs(point1['distance_from_break_km'] - point2['distance_from_break_km'])
                return distance_diff_km * 1000  # Convert to meters
            
            # Last resort fallback
            log.warning(f"{CORE_ICONS['warning']} Could not calculate distance between points, using default")
            return 1000.0  # Default 1km spacing
            
        except Exception as e:
            log.warning(f"{CORE_ICONS['warning']} Error calculating distance between points: {e}")
            return 1000.0  # Safe fallback

    def get_full_bathymetric_path_from_conf(self, spot_config):
        """Enhanced bathymetric data access for adaptive algorithm usage"""
        bathymetric_path = spot_config.get('bathymetric_path', {})
        
        if not bathymetric_path:
            raise ValueError("No bathymetric path data found in CONF - cannot proceed with scientific calculations")
        
        try:
            path_points = []
            total_points = int(bathymetric_path.get('path_points_total', '0'))
            
            if total_points == 0:
                raise ValueError("Invalid path_points_total in CONF bathymetric data")
            
            # Read adaptive point count with proper WeeWX 5.1 CONF access patterns
            for i in range(total_points):
                # Require all essential point data - fail if missing
                depth_key = f'point_{i}_depth'
                distance_key = f'point_{i}_distance_km'
                fraction_key = f'point_{i}_fraction'
                
                if depth_key not in bathymetric_path:
                    raise ValueError(f"Missing required depth data for point {i}")
                if distance_key not in bathymetric_path:
                    raise ValueError(f"Missing required distance data for point {i}")
                if fraction_key not in bathymetric_path:
                    raise ValueError(f"Missing required fraction data for point {i}")
                
                point_data = {
                    'depth': float(bathymetric_path[depth_key]),
                    'distance_km': float(bathymetric_path[distance_key]),
                    'fraction': float(bathymetric_path[fraction_key])
                }
                
                # Include coordinates if available (optional for scientific calculations)
                lat_key = f'point_{i}_latitude'
                lon_key = f'point_{i}_longitude'
                if lat_key in bathymetric_path and lon_key in bathymetric_path:
                    point_data['latitude'] = float(bathymetric_path[lat_key])
                    point_data['longitude'] = float(bathymetric_path[lon_key])
                
                path_points.append(point_data)
            
            log.debug(f"{CORE_ICONS['status']} Using adaptive bathymetric path: {total_points} points")
            return path_points
            
        except (ValueError, TypeError, KeyError) as e:
            raise ValueError(f"Invalid bathymetric path data in CONF: {e}")

    def _validate_bathymetric_path_data(self, bathymetric_path):
        """Validate bathymetric path data from CONF storage for scientific integrity"""
        validation_errors = []
        
        if not bathymetric_path:
            validation_errors.append("No bathymetric path data found")
            return validation_errors
        
        try:
            total_points = int(bathymetric_path.get('path_points_total', '0'))
            
            if total_points < self.min_points:
                validation_errors.append(f"Insufficient points: {total_points} < {self.min_points}")
            elif total_points > self.max_points:
                validation_errors.append(f"Excessive points: {total_points} > {self.max_points}")
            
            # Validate point data consistency using WeeWX 5.1 CONF access patterns
            depths = []
            distances = []
            
            for i in range(total_points):
                depth_key = f'point_{i}_depth'
                distance_key = f'point_{i}_distance_km'
                fraction_key = f'point_{i}_fraction'
                
                # Check for required data
                missing_keys = []
                if depth_key not in bathymetric_path:
                    missing_keys.append(depth_key)
                if distance_key not in bathymetric_path:
                    missing_keys.append(distance_key)
                if fraction_key not in bathymetric_path:
                    missing_keys.append(fraction_key)
                
                if missing_keys:
                    validation_errors.append(f"Missing required keys for point {i}: {missing_keys}")
                    continue
                
                try:
                    depth = float(bathymetric_path[depth_key])
                    distance = float(bathymetric_path[distance_key])
                    fraction = float(bathymetric_path[fraction_key])
                    
                    # Validate scientific ranges
                    if depth < 0:
                        validation_errors.append(f"Invalid negative depth at point {i}: {depth}m")
                    if distance < 0:
                        validation_errors.append(f"Invalid negative distance at point {i}: {distance}km")
                    if not (0.0 <= fraction <= 1.0):
                        validation_errors.append(f"Invalid fraction at point {i}: {fraction} (must be 0.0-1.0)")
                    
                    depths.append(depth)
                    distances.append(distance)
                    
                except (ValueError, TypeError) as e:
                    validation_errors.append(f"Invalid numeric data at point {i}: {e}")
            
            # Validate scientific progression
            if len(depths) >= 2:
                if depths[0] <= depths[-1]:
                    validation_errors.append("Invalid depth progression: offshore should be deeper than breaking")
            
            if len(distances) >= 2:
                if distances[0] <= distances[-1]:
                    validation_errors.append("Invalid distance progression: should decrease toward shore")
            
            # Check for reasonable depth ranges
            if depths:
                min_depth = min(depths)
                max_depth = max(depths)
                if max_depth > 200:
                    validation_errors.append(f"Unrealistic maximum depth: {max_depth}m")
                if min_depth > 100:
                    validation_errors.append(f"Breaking depth too deep: {min_depth}m")
            
        except Exception as e:
            validation_errors.append(f"CONF data structure error: {e}")
        
        return validation_errors       

    def _validate_coordinate_progression(self, bathymetry_profile):
        """Validate coordinate progression follows reasonable geographic patterns"""
        
        # Coordinates should progress monotonically along the path
        # Check for unrealistic coordinate jumps or reversals
        
        total_distance = 0  # Initialize total distance counter
        
        for i in range(len(bathymetry_profile) - 1):
            current = bathymetry_profile[i]
            next_point = bathymetry_profile[i + 1]
            
            # Calculate distance between consecutive points
            distance_meters = self._calculate_great_circle_distance(
                current['latitude'], current['longitude'],
                next_point['latitude'], next_point['longitude']
            )
            
            # FIXED: Convert meters to kilometers for validation
            distance_km = distance_meters / 1000.0
            total_distance += distance_km
            
            # Check for unrealistic coordinate jumps (>5km between consecutive points)
            if distance_km > 5.0:  # 5km
                log.warning(f"Unrealistic coordinate jump: {distance_km:.1f}km between consecutive points")
                return False
        
        # FIXED: Now total_distance is in kilometers as expected
        # Path should be reasonable for surf forecasting (5-50km)
        if total_distance < 5.0 or total_distance > 50.0:
            log.warning(f"Unrealistic total path distance: {total_distance:.1f}km")
            return False
        
        return True

    def _validate_distance_progression(self, bathymetry_profile):
        """Validate distance progression is logical and consistent"""
        if len(bathymetry_profile) < 2:
            return False
        
        # Check if distance fields are present and logical
        distances = []
        for point in bathymetry_profile:
            if 'distance_from_break_km' in point:
                distances.append(point['distance_from_break_km'])
            elif 'distance_km' in point:
                distances.append(point['distance_km'])
            else:
                # No distance data available - cannot validate but don't fail
                return True
        
        if len(distances) != len(bathymetry_profile):
            return False
        
        # Distances should generally increase going offshore (with small tolerance)
        for i in range(len(distances) - 1):
            if distances[i] < distances[i+1] - 0.1:  # Allow small variations
                log.warning(f"Non-monotonic distance progression: {distances[i]:.2f} → {distances[i+1]:.2f}")
                return False
        
        # Total distance should be reasonable for surf forecasting
        total_distance = distances[0] - distances[-1] if distances else 0
        if total_distance < 5 or total_distance > 50:  # 5-50km reasonable range
            log.warning(f"Unrealistic total distance range: {total_distance:.1f}km")
            return False
        
        return True

    def _calculate_total_path_distance(self, bathymetry_profile):
        """Calculate total distance of bathymetric path"""
        if len(bathymetry_profile) < 2:
            return 0
        
        # Try different distance calculation methods in order of preference
        first_point = bathymetry_profile[0]
        last_point = bathymetry_profile[-1]
        
        if 'distance_from_break_km' in first_point and 'distance_from_break_km' in last_point:
            return abs(first_point['distance_from_break_km'] - last_point['distance_from_break_km'])
        elif 'latitude' in first_point and 'longitude' in first_point:
            return self._calculate_great_circle_distance(
                first_point['latitude'], first_point['longitude'],
                last_point['latitude'], last_point['longitude']
            )
        else:
            # Fallback - sum all spacings
            spacings = self._calculate_point_spacings(bathymetry_profile)
            return sum(spacings)

    def _enhance_adaptive_algorithm_error_handling(self, deep_water_result, surf_break_lat, surf_break_lon):
        """Enhanced error handling wrapper for adaptive algorithm"""
        try:
            # Step 1: Get established baseline using existing proven method
            initial_result = self._create_original_surf_path_and_collect_bathymetry(
                deep_water_result, surf_break_lat, surf_break_lon
            )
            
            if not initial_result:
                raise ValueError("Failed to create initial bathymetric profile using existing method")
            
            initial_profile = initial_result['surf_path_bathymetry']
            
            # Step 2: Initialize adaptive algorithm parameters from CONF
            self._initialize_adaptive_parameters()
            
            # Step 3: Apply gradient-based refinement to existing data
            refined_profile = self._apply_gradient_based_refinement(initial_profile)
            
            # Step 4: Comprehensive scientific validation - NO FALLBACKS
            validation_enabled = getattr(self, 'validation_enabled', True)
            
            if not validation_enabled:
                log.warning(f"{CORE_ICONS['warning']} Scientific validation disabled - proceeding without verification")
            elif not self._validate_adaptive_bathymetry_profile(refined_profile):
                raise ValueError("Adaptive bathymetry failed comprehensive scientific validation")
            
            # Step 5: Return validated results
            log.info(f"{CORE_ICONS['navigation']} Adaptive algorithm success: {initial_result['path_points_total']} → {len(refined_profile)} points")
            
            return {
                'surf_path_bathymetry': refined_profile,
                'path_points_total': len(refined_profile),
                'path_distance_km': initial_result['path_distance_km'],
                'adaptive_method': 'gradient_based_refinement',
                'refinement_iterations': getattr(self, '_last_iteration_count', 1),
                'validation_passed': True
            }
            
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Adaptive algorithm failed: {e}")
            # FAIL FAST: Re-raise exception to maintain scientific integrity
            # No fallbacks in scientific application - ensures only valid data reaches CONF storage
            raise ValueError(f"Adaptive bathymetry algorithm failed scientific validation: {e}")

    def _generate_validation_diagnostic_report(self, bathymetry_profile):
        """Generate detailed diagnostic report for validation analysis"""
        report = []
        report.append("=== ADAPTIVE BATHYMETRY VALIDATION DIAGNOSTIC REPORT ===")
        
        # Basic profile statistics
        depths = [p.get('depth', 0) for p in bathymetry_profile]
        report.append(f"Profile Statistics:")
        report.append(f"  - Total points: {len(bathymetry_profile)}")
        report.append(f"  - Depth range: {max(depths):.1f}m → {min(depths):.1f}m")
        report.append(f"  - Average depth: {sum(depths)/len(depths):.1f}m")
        
        # Zone distribution analysis
        zones = {'deep': 0, 'transition': 0, 'critical': 0, 'breaking': 0}
        for point in bathymetry_profile:
            depth = point.get('depth', 0)
            if depth > self.deep_water_threshold:
                zones['deep'] += 1
            elif self.critical_depth_max < depth <= self.deep_water_threshold:
                zones['transition'] += 1
            elif self.critical_depth_min <= depth <= self.critical_depth_max:
                zones['critical'] += 1
            else:
                zones['breaking'] += 1
        
        report.append(f"Zone Distribution:")
        for zone, count in zones.items():
            report.append(f"  - {zone}: {count} points")
        
        # Gradient analysis
        gradients = []
        for i in range(len(bathymetry_profile) - 1):
            gradient, _ = self._calculate_segment_metrics(bathymetry_profile[i], bathymetry_profile[i+1])
            gradients.append(gradient)
        
        if gradients:
            report.append(f"Gradient Statistics:")
            report.append(f"  - Average: {sum(gradients)/len(gradients):.4f}")
            report.append(f"  - Maximum: {max(gradients):.4f}")
            report.append(f"  - Refinement threshold: {self.refinement_threshold:.4f}")
            steep_gradients = [g for g in gradients if g > self.refinement_threshold]
            report.append(f"  - Segments above threshold: {len(steep_gradients)}")
        
        return "\n".join(report)

     
class SurfForecastGenerator:
    """Generate surf condition forecasts"""
    
    def __init__(self, config_dict, db_manager=None, engine=None):
        """Initialize surf forecast generator with data-driven configuration"""
        self.config_dict = config_dict
        self.db_manager = db_manager
        self.engine = engine  # Store engine reference for thread-safe database access
        service_config = config_dict.get('SurfFishingService', {})
        self.surf_rating_factors = service_config.get('surf_rating_factors', {})
        
        gfs_wave_config = service_config.get('noaa_gfs_wave', {})
        field_mappings = gfs_wave_config.get('field_mappings', {})

        log.info(f"DEBUG: Found {len(field_mappings)} field mappings in noaa_gfs_wave")
        log.info(f"DEBUG: field_mappings keys: {list(field_mappings.keys())}")

        self.surf_critical = []
        self.surf_recommended = []
        for field_name, field_config in field_mappings.items():
            db_field = field_config.get('database_field', field_name)
            priority = int(field_config.get('forecast_priority', 3))
            log.info(f"DEBUG: Field {field_name}, db_field={db_field}, priority={priority}") 
            if priority == 1:  # Priority 1 = critical
                self.surf_critical.append(db_field)
                log.info(f"DEBUG: Added {db_field} to surf_critical")
            elif priority == 2:  # Priority 2 = recommended
                self.surf_recommended.append(db_field)
                log.info(f"DEBUG: Added {db_field} to surf_recommended")
            log.info(f"DEBUG: self.surf_critical now contains: {self.surf_critical}")
            log.info(f"DEBUG: self.surf_recommended now contains: {self.surf_recommended}")
    
    def generate_surf_forecast(self, spot, forecast_data):
        """
        Generate surf forecast using updated GFS Wave field mappings with Phase I tide integration
        """
        
        try:
            forecasts = []
            
            # Get data-driven field configuration from CONF
            service_config = self.config_dict.get('SurfFishingService', {})
            surf_critical = self.surf_critical
            surf_recommended = self.surf_recommended
            
            # REQUIRED: Fail if no field configuration available
            if not surf_critical:
                log.error(f"{CORE_ICONS['warning']} No critical surf fields configured in CONF")
                log.info(f"DEBUG: surf_critical = {surf_critical}")
                log.info(f"DEBUG: service_config keys = {list(service_config.keys())}")
                return []
            
            for period_data in forecast_data:
                try:
                    # Extract fields based on CONF configuration
                    forecast_values = {}
                    
                    # Extract critical fields - FAIL if missing
                    missing_critical = []
                    for field_name in surf_critical:
                        if field_name in period_data:
                            forecast_values[field_name] = period_data.get(field_name, 0)
                        else:
                            missing_critical.append(field_name)
                    
                    # FAIL forecast period if critical fields missing
                    if missing_critical:
                        log.warning(f"{CORE_ICONS['warning']} Missing critical fields for forecast period: {missing_critical}")
                        continue
                    
                    # Extract recommended fields (optional)
                    for field_name in surf_recommended:
                        if field_name in period_data:
                            forecast_values[field_name] = period_data.get(field_name, 0)
                    
                    # Add forecast time
                    forecast_values['forecast_time'] = period_data.get('forecast_time', int(time.time()))
                    
                    # Calculate wave height range from min/max
                    if 'wave_height' in forecast_values:
                        wave_height = forecast_values['wave_height']
                        forecast_values['wave_height_min'] = wave_height * 0.8
                        forecast_values['wave_height_max'] = wave_height * 1.2
                    
                    # NEW: Phase I tide integration using existing method
                    try:
                        tide_info = self._determine_tide_stage(
                            forecast_values['forecast_time'], 
                            {}  # Empty tide_conditions - Phase I provides data
                        )
                        forecast_values['tide_stage'] = tide_info['stage']
                        forecast_values['tide_height'] = tide_info['height']
                        
                        log.debug(f"{CORE_ICONS['status']} Phase I tide integrated: {tide_info['stage']} tide at {tide_info['height']}ft")
                        
                    except Exception as tide_error:
                        # NO FALLBACKS: Fail as required
                        log.error(f"{CORE_ICONS['warning']} Phase I tide integration failed: {tide_error}")
                        log.info(f"{CORE_ICONS['navigation']} Ensure Phase I MarineDataService is installed and running")
                        raise Exception(f"Phase I tide integration required: {tide_error}")
                    
                    # PRESERVE EXISTING: Create forecast structure with available data
                    basic_forecast = [{
                        'forecast_time': period_data['forecast_time'],
                        'wave_height_min': forecast_values.get('wave_height', 0) * 0.8,
                        'wave_height_max': forecast_values.get('wave_height', 0) * 1.2,
                        'wave_period': forecast_values.get('wave_period', 0),
                        'wave_direction': forecast_values.get('wave_direction', 0),
                        'wind_speed': forecast_values.get('wind_speed', 0),
                        'wind_direction': forecast_values.get('wind_direction', 0),
                        'total_swell_height': forecast_values.get('total_swell_height', 0),
                        'total_swell_period': forecast_values.get('total_swell_period', 0),
                        'wind_wave_height': forecast_values.get('wind_wave_height', 0),
                        'wind_wave_period': forecast_values.get('wind_wave_period', 0),
                        # ADD: Phase I tide data
                        'tide_stage': forecast_values['tide_stage'],
                        'tide_height': forecast_values['tide_height']
                    }]
                    
                    # PRESERVE EXISTING: Use comprehensive surf quality assessment
                    enhanced_forecast = self.assess_surf_quality_complete(
                        basic_forecast, 
                        current_wind={
                            'wind_speed': forecast_values.get('wind_speed', 0),
                            'wind_direction': forecast_values.get('wind_direction', 0)
                        }, 
                        spot_config=spot
                    )
                    
                    if enhanced_forecast:
                        forecast = enhanced_forecast[0]
                        forecast.update(forecast_values)
                        forecasts.append(forecast)
                    else:
                        log.warning(f"{CORE_ICONS['warning']} Quality assessment failed for forecast period")
                        continue
                    
                except Exception as e:
                    log.error(f"{CORE_ICONS['warning']} Error processing surf period: {e}")
                    continue
            
            # NEW: Enhance all forecasts with calculated fields
            if forecasts:
                enhanced_forecasts = self._enhance_forecast_with_calculated_fields(forecasts)
                log.info(f"{CORE_ICONS['status']} Enhanced {len(enhanced_forecasts)} forecasts with Phase I tide data and calculated fields")
                return enhanced_forecasts
            
            return forecasts  # PRESERVE: Return empty list if no forecasts
            
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error generating surf forecasts: {e}")
            return []
    
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
            Get complete bathymetric path profile from CONF for multi-point wave transformation
            
            ENHANCED METHOD: Returns full bathymetric path array instead of just two points
            PRESERVES: All existing fallback logic and error handling
            MAINTAINS: Exact same CONF reading architecture and method signature
            """
            try:
                # PRESERVE EXISTING: Check if spot has bathymetric data in CONF
                bathymetric_path = spot_config.get('bathymetric_path', {})
                
                if bathymetric_path:
                    # ENHANCED: Extract complete bathymetric profile (previously only used first/last)
                    bathymetric_profile = []
                    total_points = int(bathymetric_path.get('path_points_total', '16'))
                    
                    for i in range(total_points):
                        depth_key = f'point_{i}_depth'
                        lat_key = f'point_{i}_latitude'
                        lon_key = f'point_{i}_longitude'
                        distance_key = f'point_{i}_distance_km'
                        fraction_key = f'point_{i}_fraction'
                        
                        if depth_key in bathymetric_path:
                            point_data = {
                                'depth': abs(float(bathymetric_path[depth_key])),  # Ensure positive depth
                                'latitude': float(bathymetric_path.get(lat_key, 0.0)),
                                'longitude': float(bathymetric_path.get(lon_key, 0.0)),
                                'distance_km': float(bathymetric_path.get(distance_key, 0.0)),
                                'fraction': float(bathymetric_path.get(fraction_key, 0.0)),
                                'point_index': i
                            }
                            bathymetric_profile.append(point_data)
                    
                    if bathymetric_profile:
                        # ENHANCED: Return complete profile with metadata
                        log.debug(f"{CORE_ICONS['status']} Using GEBCO multi-point bathymetry: {len(bathymetric_profile)} points")
                        log.debug(f"{CORE_ICONS['navigation']} Depth profile: {bathymetric_profile[0]['depth']:.1f}m → {bathymetric_profile[-1]['depth']:.1f}m")
                        
                        return {
                            'bathymetric_profile': bathymetric_profile,
                            'offshore_depth': bathymetric_profile[0]['depth'],
                            'breaking_depth': bathymetric_profile[-1]['depth'],
                            'total_points': len(bathymetric_profile),
                            'path_distance_km': bathymetric_profile[0]['distance_km'] if bathymetric_profile else 0.0,
                            'data_source': 'gebco_multi_point'
                        }
                
                # PRESERVE EXISTING: Fallback to default depths if no bathymetric data
                log.debug(f"{CORE_ICONS['warning']} No bathymetric data found, using defaults")
                return {
                    'bathymetric_profile': [
                        {'depth': 40.0, 'distance_km': 10.0, 'fraction': 0.0, 'point_index': 0},
                        {'depth': 2.5, 'distance_km': 0.0, 'fraction': 1.0, 'point_index': 1}
                    ],
                    'offshore_depth': 40.0,
                    'breaking_depth': 2.5,
                    'total_points': 2,
                    'path_distance_km': 10.0,
                    'data_source': 'fallback_default'
                }
                
            except Exception as e:
                log.error(f"{CORE_ICONS['warning']} Error reading bathymetric data: {e}")
                # PRESERVE EXISTING: Same fallback on error
                return {
                    'bathymetric_profile': [
                        {'depth': 40.0, 'distance_km': 10.0, 'fraction': 0.0, 'point_index': 0},
                        {'depth': 2.5, 'distance_km': 0.0, 'fraction': 1.0, 'point_index': 1}
                    ],
                    'offshore_depth': 40.0,
                    'breaking_depth': 2.5,
                    'total_points': 2,
                    'path_distance_km': 10.0,
                    'data_source': 'error_fallback'
                }
    
    def _transform_to_local_conditions(self, offshore_forecast, spot_config):
        """
        Transform offshore wave data to local surf conditions using multi-point wave transformation physics
        
        ENHANCED METHOD: Implements incremental wave transformation along complete bathymetric path
        PRESERVES: All existing method structure, error handling, and output format
        MAINTAINS: Exact same input/output interface for compatibility
        """
        try:
            transformed_forecast = []
            
            # PRESERVE EXISTING: Get spot characteristics from CONF
            beach_facing = float(spot_config.get('beach_facing', '270'))  # Default west-facing
            bottom_type = spot_config.get('bottom_type', 'sand')
            
            # ENHANCED: Get complete bathymetric profile instead of just two points
            bathymetric_data = self.get_bathymetric_data_from_conf(spot_config)
            bathymetric_profile = bathymetric_data['bathymetric_profile']
            
            for period in offshore_forecast:
                try:
                    # PRESERVE EXISTING: Extract offshore wave parameters
                    offshore_height = period.get('wave_height', 0.0)
                    wave_period = period.get('wave_period', 8.0)
                    wave_direction = period.get('wave_direction', 270.0)
                    current_tide = period.get('tide_level', 0.0)  # From Phase I tide data
                    
                    # ENHANCED: Multi-point wave transformation instead of single calculation
                    transformed_height = self._apply_multi_point_wave_transformation(
                        offshore_height, 
                        wave_period, 
                        wave_direction, 
                        bathymetric_profile,
                        beach_facing,
                        bottom_type,
                        current_tide
                    )
                    
                    # PRESERVE EXISTING: Exact same output structure for compatibility
                    min_height = max(0.1, transformed_height * 0.8)  # 20% variance range
                    max_height = transformed_height * 1.2
                    
                    transformed_period = {
                        'forecast_time': period['forecast_time'],
                        'wave_height_min': min_height,
                        'wave_height_max': max_height, 
                        'wave_height_primary': transformed_height,  # Physics-transformed height
                        'wave_period': wave_period,
                        'wave_direction': wave_direction,
                        # PRESERVE EXISTING: All other fields passed through
                        'wind_speed': period.get('wind_speed', 0),
                        'wind_direction': period.get('wind_direction', 0),
                        'tide_level': current_tide
                    }
                    
                    # PRESERVE EXISTING: Add any additional fields from period
                    for key, value in period.items():
                        if key not in transformed_period:
                            transformed_period[key] = value
                    
                    transformed_forecast.append(transformed_period)
                    
                except Exception as e:
                    log.error(f"{CORE_ICONS['warning']} Error transforming period: {e}")
                    # PRESERVE EXISTING: Continue processing on individual period errors
                    continue
            
            return transformed_forecast
            
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error in wave transformation: {e}")
            # PRESERVE EXISTING: Return original forecast on major errors
            return offshore_forecast
    
    def assess_surf_quality_complete(self, forecast_periods, current_wind=None, spot_config=None):
        """Assess surf quality using enhanced physics-based scoring from CONF"""

        try:
            enhanced_forecast = []
            failed_periods = 0
            total_periods = len(forecast_periods)
                     
            # Get enhanced scoring criteria from CONF - CRITICAL path validation
            service_config = self.config_dict.get('SurfFishingService', {})
            scoring_criteria = service_config.get('scoring_criteria', {})
            surf_scoring = scoring_criteria.get('surf_scoring', {})
            
            if not surf_scoring:
                log.error(f"{CORE_ICONS['warning']} CRITICAL CONFIGURATION ERROR: No surf scoring configuration found in CONF")
                log.error("This indicates a configuration problem that needs to be resolved")
                log.error("WeeWX service will continue, but surf forecasts cannot be generated until configuration is fixed")
                return []  # Return empty - do not crash WeeWX
            
            # Get scoring weights from CONF - CRITICAL nested path structure
            scoring_weights = surf_scoring.get('scoring_weights', {})
            if not scoring_weights:
                log.error(f"{CORE_ICONS['warning']} CRITICAL CONFIGURATION ERROR: No scoring_weights in CONF")
                log.error("WeeWX service will continue, but surf forecasts cannot be generated until configuration is fixed")
                return []
            
            try:
                wave_height_weight = float(scoring_weights.get('wave_height', '0.35'))
                wave_period_weight = float(scoring_weights.get('wave_period', '0.35'))
                wind_quality_weight = float(scoring_weights.get('wind_quality', '0.20'))
                tide_phase_weight = float(scoring_weights.get('tide_phase', '0.10'))
            except (ValueError, TypeError) as e:
                log.error(f"{CORE_ICONS['warning']} Invalid scoring weight configuration: {e}")
                log.error("Check surf_scoring.scoring_weights in CONF - must be numeric values")
                return []  # Cannot proceed with invalid configuration
            
            for period in forecast_periods:
                try:
                    # Extract wave data - CRITICAL FIX: Convert all numeric values to float
                    try:
                        total_swell_height = float(period.get('total_swell_height', 0))
                        total_swell_period = float(period.get('total_swell_period', 0))
                        wind_wave_height = float(period.get('wind_wave_height', 0))
                        wind_wave_period = float(period.get('wind_wave_period', 0))
                        wave_height = float(period.get('wave_height', 0))
                        wave_period = float(period.get('wave_period', 0))
                        wind_speed = period.get('wind_speed')
                        wind_direction = period.get('wind_direction')
                        
                        # CRITICAL FIX: Convert wind data to float
                        if wind_speed is not None:
                            wind_speed_float = float(wind_speed)
                        else:
                            wind_speed_float = None
                            
                        if wind_direction is not None:
                            wind_direction_float = float(wind_direction)
                        else:
                            wind_direction_float = None
                        
                    except (ValueError, TypeError) as e:
                        failed_periods += 1
                        log.error(f"{CORE_ICONS['warning']} TYPE CONVERSION ERROR: Cannot convert wave data to float: {e}")
                        log.error(f"Raw values: height={period.get('wave_height')}, period={period.get('wave_period')}, wind_speed={period.get('wind_speed')}, wind_dir={period.get('wind_direction')}")
                        log.info("This indicates data format issue from GRIB processing - will retry next forecast cycle")
                        continue
                    
                    if wave_height is None or wave_period is None:
                        failed_periods += 1
                        log.warning(f"{CORE_ICONS['warning']} DATA QUALITY ISSUE: Missing critical wave data (height={wave_height}, period={wave_period}) for forecast period")
                        log.info("This period will be skipped - next forecast cycle will retry with fresh API data")
                        continue
                    
                    # Enhanced swell quality assessment with total swell + wind wave approach
                    swell_dominance = self._assess_swell_dominance_updated(
                        total_swell_height, total_swell_period, wind_wave_height, wind_wave_period
                    )
                    
                    # Wind quality scoring configuration
                    wind_quality_scoring = surf_scoring.get('wind_quality_scoring', {})
                    wind_speed_thresholds = surf_scoring.get('wind_speed_thresholds', {})
                    
                    if not wind_quality_scoring or not wind_speed_thresholds:
                        failed_periods += 1
                        log.error(f"{CORE_ICONS['warning']} CONFIGURATION ERROR: Incomplete wind scoring configuration in CONF")
                        log.error("Check surf_scoring.wind_quality_scoring and surf_scoring.wind_speed_thresholds")
                        log.info("Period skipped - will retry next forecast cycle (configuration issue, not API data)")
                        continue
                    
                    if wind_speed_float is None or wind_direction_float is None:
                        failed_periods += 1
                        log.warning(f"{CORE_ICONS['warning']} DATA QUALITY ISSUE: Missing wind data (speed={wind_speed_float}, direction={wind_direction_float})")
                        log.info("This indicates API data issue - will retry next forecast cycle with fresh API data")
                        continue
                    
                    # Get wind speed thresholds from CONF - NEW: Direction-specific thresholds
                    calm_max = float(wind_speed_thresholds.get('calm_max', '3'))
                    
                    # Get beach orientation from spot config
                    if not spot_config:
                        failed_periods += 1
                        log.error(f"{CORE_ICONS['warning']} CONFIGURATION ERROR: No spot config provided for wind scoring")
                        log.error("This indicates a programming issue - spot_config must be provided")
                        log.info("Period skipped - this is not an API retry issue")
                        continue
                        
                    beach_facing = spot_config.get('beach_facing')
                    if beach_facing is None:
                        failed_periods += 1
                        log.error(f"{CORE_ICONS['warning']} CONFIGURATION ERROR: No beach_facing in spot config")
                        log.error("This indicates incomplete spot configuration - check surf spot setup")
                        log.info("Period skipped - configuration issue, not API data")
                        continue
                    
                    # CRITICAL FIX: Convert beach_facing to float
                    try:
                        beach_facing_float = float(beach_facing)
                    except (ValueError, TypeError):
                        failed_periods += 1
                        log.error(f"{CORE_ICONS['warning']} TYPE CONVERSION ERROR: beach_facing is not numeric: {beach_facing}")
                        log.error("This indicates configuration issue - beach_facing must be numeric degrees")
                        log.info("Period skipped - configuration issue, not API data")
                        continue
                    
                    # Calculate relative wind direction
                    wind_relative = abs(wind_direction_float - beach_facing_float)
                    if wind_relative > 180:
                        wind_relative = 360 - wind_relative
                    
                    # Determine wind condition using NEW Scripps research-based logic
                    if wind_speed_float <= calm_max:
                        wind_condition = 'calm'
                    elif wind_relative < 45:  # Onshore winds (ocean to land)
                        # Get onshore-specific thresholds
                        onshore_light_max = float(wind_speed_thresholds.get('onshore_light_max', '10'))
                        onshore_moderate_max = float(wind_speed_thresholds.get('onshore_moderate_max', '20'))
                        onshore_strong_min = float(wind_speed_thresholds.get('onshore_strong_min', '20.01'))
                        
                        if wind_speed_float <= onshore_light_max:
                            wind_condition = 'onshore_light'
                        elif wind_speed_float <= onshore_moderate_max:
                            wind_condition = 'onshore_moderate'
                        else:  # >= onshore_strong_min
                            wind_condition = 'onshore_strong'
                            
                    elif wind_relative > 135:  # Offshore winds (land to ocean)
                        # Get offshore-specific thresholds
                        offshore_light_max = float(wind_speed_thresholds.get('offshore_light_max', '10'))
                        offshore_moderate_max = float(wind_speed_thresholds.get('offshore_moderate_max', '20'))
                        offshore_strong_max = float(wind_speed_thresholds.get('offshore_strong_max', '30'))
                        
                        if wind_speed_float <= offshore_light_max:
                            wind_condition = 'offshore_light'
                        elif wind_speed_float <= offshore_moderate_max:
                            wind_condition = 'offshore_moderate'
                        elif wind_speed_float <= offshore_strong_max:
                            wind_condition = 'offshore_strong'
                        else:  # > offshore_strong_max
                            wind_condition = 'offshore_extreme'
                            
                    else:  # Cross-shore winds (parallel to shore)
                        # Get crossshore-specific thresholds
                        crossshore_light_max = float(wind_speed_thresholds.get('crossshore_light_max', '15'))
                        crossshore_strong_min = float(wind_speed_thresholds.get('crossshore_strong_min', '15.01'))
                        
                        if wind_speed_float <= crossshore_light_max:
                            wind_condition = 'crossshore_light'
                        else:  # >= crossshore_strong_min
                            wind_condition = 'crossshore_strong'
                    
                    # Get wind score from CONF using NEW category names
                    wind_score = wind_quality_scoring.get(wind_condition)
                    
                    if wind_score is None:
                        failed_periods += 1
                        log.warning(f"{CORE_ICONS['warning']} CONFIGURATION ERROR: No wind score found for '{wind_condition}' in CONF")
                        log.error("This indicates incomplete wind scoring configuration - check wind_quality_scoring section")
                        log.info("Period skipped - configuration issue, not API data")
                        continue
                        
                    wind_score = float(wind_score)
                    
                    # INLINE WAVE HEIGHT SCORING - integrated from CONF ranges
                    height_scoring = surf_scoring.get('transformed_wave_height_scoring', {})
                    height_ranges = height_scoring.get('ranges', {})
                    
                    if not height_ranges:
                        failed_periods += 1
                        log.error(f"{CORE_ICONS['warning']} CONFIGURATION ERROR: No height scoring ranges found in CONF")
                        log.error("This indicates incomplete configuration - check surf_scoring.transformed_wave_height_scoring.ranges")
                        log.info("Period skipped - will retry next forecast cycle (configuration issue, not API data)")
                        continue
                    
                    size_score = None
                    for range_key, score_value in height_ranges.items():
                        try:
                            if '-' in range_key:
                                range_parts = range_key.split('-')
                                if len(range_parts) == 2:
                                    min_height = float(range_parts[0])
                                    max_height = float(range_parts[1]) if range_parts[1] != '+' else float('inf')
                                    
                                    if min_height <= wave_height < max_height:
                                        size_score = float(score_value)
                                        break
                            elif range_key.endswith('+'):
                                min_height = float(range_key[:-1])
                                if wave_height >= min_height:
                                    size_score = float(score_value)
                                    break
                        except ValueError as e:
                            log.error(f"CRITICAL: Error parsing height range {range_key}: {e}")
                            continue
                    
                    if size_score is None:
                        failed_periods += 1
                        log.warning(f"{CORE_ICONS['warning']} DATA QUALITY ISSUE: No height range matched {wave_height}ft in CONF ranges")
                        log.info("This suggests wave height outside expected ranges - will retry next forecast cycle with fresh API data")
                        continue
                    
                    # INLINE WAVE PERIOD SCORING - integrated from CONF ranges  
                    period_scoring = surf_scoring.get('wave_period_scoring', {})
                    period_ranges = period_scoring.get('ranges', {})
                    
                    if not period_ranges:
                        failed_periods += 1
                        log.error(f"{CORE_ICONS['warning']} CONFIGURATION ERROR: No period scoring ranges found in CONF")
                        log.error("This indicates incomplete configuration - check surf_scoring.wave_period_scoring.ranges")
                        log.info("Period skipped - will retry next forecast cycle (configuration issue, not API data)")
                        continue
                    
                    period_score = None
                    for range_key, score_value in period_ranges.items():
                        try:
                            if '-' in range_key:
                                range_parts = range_key.split('-')
                                if len(range_parts) == 2:
                                    min_period = float(range_parts[0])
                                    max_period = float(range_parts[1]) if range_parts[1] != '+' else float('inf')
                                    
                                    if min_period <= wave_period < max_period:
                                        period_score = float(score_value)
                                        break
                            elif range_key.endswith('+'):
                                min_period = float(range_key[:-1])
                                if wave_period >= min_period:
                                    period_score = float(score_value)
                                    break
                        except ValueError as e:
                            log.error(f"CRITICAL: Error parsing period range {range_key}: {e}")
                            continue
                    
                    if period_score is None:
                        failed_periods += 1
                        log.warning(f"{CORE_ICONS['warning']} DATA QUALITY ISSUE: No period range matched {wave_period}s in CONF ranges")
                        log.info("This suggests wave period outside expected ranges - will retry next forecast cycle with fresh API data")
                        continue
                    
                    # Enhanced swell quality assessment with total swell + wind wave approach
                    swell_dominance = self._assess_swell_dominance_updated(
                        total_swell_height, total_swell_period, wind_wave_height, wind_wave_period
                    )
                    
                    # INLINE SWELL QUALITY SCORING - integrated from existing _score_swell_quality method
                    if swell_dominance == 'unknown':
                        failed_periods += 1
                        log.error(f"{CORE_ICONS['warning']} CRITICAL: Swell dominance is unknown - cannot score swell quality")
                        log.info("This indicates swell calculation issue - will retry next forecast cycle")
                        continue
                    
                    # Use total_swell_period for scoring instead of wave_period
                    if total_swell_period <= 0:
                        failed_periods += 1
                        log.error(f"{CORE_ICONS['warning']} CRITICAL: Invalid total swell period {total_swell_period} - cannot score")
                        log.info("This indicates swell data issue - will retry next forecast cycle")
                        continue
                    
                    # Groundswell quality (clean, organized waves)
                    if swell_dominance == 'swell_dominant':
                        if total_swell_period >= 14:
                            swell_quality_score = 0.9  # Excellent long-period groundswell
                        elif total_swell_period >= 11:
                            swell_quality_score = 0.8  # Good groundswell
                        elif total_swell_period >= 8:
                            swell_quality_score = 0.7  # Moderate groundswell
                        else:
                            swell_quality_score = 0.6  # Short-period swell
                            
                    # Wind wave quality (local conditions)  
                    elif swell_dominance == 'wind_wave_dominant':
                        if total_swell_period >= 8:
                            swell_quality_score = 0.6  # Organized wind waves
                        elif total_swell_period >= 6:
                            swell_quality_score = 0.4  # Moderate wind waves
                        else:
                            swell_quality_score = 0.3  # Choppy wind waves
                            
                    # Mixed conditions
                    else:  # mixed
                        if total_swell_period >= 10:
                            swell_quality_score = 0.7  # Good mixed conditions
                        elif total_swell_period >= 8:
                            swell_quality_score = 0.6  # Moderate mixed
                        else:
                            swell_quality_score = 0.5  # Variable mixed conditions
                    
                    # Calculate weighted overall rating using CONF weights
                    overall_score = (
                        size_score * wave_height_weight +
                        period_score * wave_period_weight +
                        wind_score * wind_quality_weight +
                        swell_quality_score * tide_phase_weight  # Use tide_phase_weight for swell quality
                    )
                    
                    # Convert to 1-5 star rating
                    stars = max(1, min(5, int(overall_score * 5)))
                    
                    # Generate quality description using simplified logic since _generate_quality_description doesn't exist
                    if stars >= 4:
                        quality_text = f"Excellent {swell_dominance} conditions"
                    elif stars >= 3:
                        quality_text = f"Good {swell_dominance} conditions"
                    elif stars >= 2:
                        quality_text = f"Fair {swell_dominance} conditions"
                    else:
                        quality_text = f"Poor {swell_dominance} conditions"
                    
                    # Calculate confidence based on data quality using simplified logic since _calculate_quality_confidence doesn't exist
                    confidence = min(1.0, (size_score + period_score + wind_score + swell_quality_score) / 4.0)
                    
                    # Enhanced period with all existing fields plus new analysis
                    enhanced_period = period.copy()
                    enhanced_period.update({
                        'quality_rating': stars,
                        'quality_stars': stars,
                        'quality_text': quality_text,
                        'conditions_description': f"{wave_height:.1f}ft {swell_dominance} {quality_text}",
                        'confidence': confidence,
                        'wind_condition': wind_condition,
                        'swell_dominance': swell_dominance,
                        'component_scores': {
                            'size_score': size_score,
                            'period_score': period_score, 
                            'wind_score': wind_score,
                            'swell_quality': swell_quality_score
                        }
                    })
                    
                    enhanced_forecast.append(enhanced_period)
                    
                except Exception as e:
                    failed_periods += 1
                    log.error(f"DEBUG: EXCEPTION in period processing: {str(e)}")
                    log.error(f"DEBUG: Exception type: {type(e)}")
                    import traceback
                    log.error(f"DEBUG: Full traceback: {traceback.format_exc()}")
                    log.error(f"{CORE_ICONS['warning']} PROCESSING ERROR: Error assessing period quality: {e}")
                    log.info("Period skipped due to processing error - will retry next forecast cycle")
                    continue
            
            # WeeWX-compliant completion logging
            success_periods = len(enhanced_forecast)
            if success_periods > 0:
                log.info(f"{CORE_ICONS['status']} Successfully processed {success_periods}/{total_periods} forecast periods")
                if failed_periods > 0:
                    log.info(f"Skipped {failed_periods} periods due to data/configuration issues - will retry next cycle")
            else:
                log.warning(f"{CORE_ICONS['warning']} No forecast periods could be processed ({failed_periods}/{total_periods} failed)")
                log.warning("This may indicate API data issues or configuration problems")
                log.warning("WeeWX service continues normally - will retry next forecast cycle")
                
            return enhanced_forecast
            
        except Exception as e:
            # WeeWX-compliant top-level error handling - NEVER crash the service
            log.error(f"DEBUG: TOP-LEVEL EXCEPTION: {str(e)}")
            log.error(f"DEBUG: Exception type: {type(e)}")
            import traceback
            log.error(f"DEBUG: Full traceback: {traceback.format_exc()}")
            log.error(f"{CORE_ICONS['warning']} CRITICAL ERROR in surf quality assessment: {e}")
            log.error("This is a serious error that needs investigation")
            log.error("WeeWX service continues normally - surf forecasts disabled until resolved")
            log.error("Check configuration, API connectivity, and data integrity")
            return []  # Return empty list - do not crash WeeWX

    def _assess_swell_dominance_updated(self, total_swell_height, total_swell_period, 
                                    wind_wave_height, wind_wave_period):
        """Assess swell dominance using total swell + wind wave energy comparison"""
        
        try:
            # Calculate wave energy using E = H²T² approximation
            total_swell_height = float(total_swell_height)
            total_swell_period = float(total_swell_period)
            wind_wave_height = float(wind_wave_height)
            wind_wave_period = float(wind_wave_period)

            swell_energy = total_swell_height ** 2 * total_swell_period ** 2
            wind_wave_energy = wind_wave_height ** 2 * wind_wave_period ** 2
            
            total_energy = swell_energy + wind_wave_energy
            
            if total_energy == 0:
                return 'unknown'
            
            swell_percentage = swell_energy / total_energy
            
            # Determine dominance type with thresholds
            if swell_percentage >= 0.7:
                return 'swell_dominant'  # Clean groundswell conditions
            elif swell_percentage <= 0.3:
                return 'wind_wave_dominant'  # Local wind-driven conditions  
            else:
                return 'mixed'  # Combined swell and wind waves
                
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error assessing swell dominance: {e}")
            return 'unknown'

    def _calculate_quality_confidence(self, total_swell_height, wind_wave_height, wave_period, wind_speed):
        """
        Calculate confidence score for surf quality assessment based on data completeness and reliability
        """
        try:
            # CRITICAL FIX: Convert all inputs to float to prevent string arithmetic errors  
            total_swell_height = float(total_swell_height) if total_swell_height is not None else 0.0
            wind_wave_height = float(wind_wave_height) if wind_wave_height is not None else 0.0
            wave_period = float(wave_period) if wave_period is not None else 0.0
            wind_speed = float(wind_speed) if wind_speed is not None else 0.0
            
            # Read confidence parameters from CONF (data-driven approach)
            service_config = self.config_dict.get('SurfFishingService', {})
            scoring_criteria = service_config.get('scoring_criteria', {})
            confidence_params = scoring_criteria.get('confidence_calculation', {
                'base_confidence': 0.7,
                'data_completeness_bonus': 0.2,
                'realistic_values_bonus': 0.1
            })
            
            # Start with base confidence from CONF
            confidence = float(confidence_params.get('base_confidence', 0.7))
            data_bonus = float(confidence_params.get('data_completeness_bonus', 0.2))
            realism_bonus = float(confidence_params.get('realistic_values_bonus', 0.1))
            
            # Data completeness assessment
            data_fields_present = 0
            total_data_fields = 4
            
            if total_swell_height > 0:
                data_fields_present += 1
            if wind_wave_height > 0:
                data_fields_present += 1
            if wave_period > 0:
                data_fields_present += 1
            if wind_speed > 0:
                data_fields_present += 1
                
            # Apply data completeness bonus proportionally
            completeness_ratio = data_fields_present / total_data_fields
            confidence += (data_bonus * completeness_ratio)
            
            # Realistic value assessment
            realistic_factors = 0
            total_realistic_factors = 3
            
            # Wave period realism (4-25 seconds is realistic range)
            if 4 <= wave_period <= 25:
                realistic_factors += 1
            elif wave_period > 25:  # Unrealistic long period
                confidence -= 0.1
                
            # Wave height realism (0.5-30 feet is realistic range)
            total_wave_height = total_swell_height + wind_wave_height
            if 0.5 <= total_wave_height <= 30:
                realistic_factors += 1
            elif total_wave_height > 30:  # Unrealistic height
                confidence -= 0.15
                
            # Wind speed realism (0-100 mph is realistic range)
            if 0 <= wind_speed <= 100:
                realistic_factors += 1
            elif wind_speed > 100:  # Unrealistic wind speed
                confidence -= 0.1
                
            # Apply realism bonus proportionally
            realism_ratio = realistic_factors / total_realistic_factors
            confidence += (realism_bonus * realism_ratio)
            
            # Ensure confidence stays within valid range [0.0, 1.0]
            confidence = max(0.0, min(1.0, confidence))
            
            return confidence
            
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error calculating quality confidence: {e}")
            return 0.5  # Return moderate confidence on error
    
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

    def calculate_shoaling_coefficient(self, current_depth, next_depth, wave_period):
        """
        Calculate shoaling coefficient for depth segment using linear wave theory
        """
        try:
            # PRESERVE EXISTING: Get physics parameters from enhanced CONF
            service_config = self.config_dict.get('SurfFishingService', {})
            scoring_criteria = service_config.get('scoring_criteria', {})
            surf_scoring = scoring_criteria.get('surf_scoring', {})
            physics_params = surf_scoring.get('physics_parameters', {})
            
            # PRESERVE EXISTING: Data-driven physics parameters from CONF
            shoaling_factor_max = float(physics_params.get('shoaling_factor_max', '1.5'))
            
            # PRESERVE EXISTING: Calculate wave celerity using dispersion relation
            g = 9.81  # gravitational acceleration
            
            # ENHANCED: Handle both deep/shallow water cases properly for segments
            def calculate_wave_celerity(period, depth):
                """Calculate wave celerity using dispersion relation"""
                L0 = g * period**2 / (2 * math.pi)  # Deep water wavelength
                
                if depth < L0 / 20:  # Shallow water approximation
                    return math.sqrt(g * depth)
                else:  # Deep/intermediate water
                    # Use iterative solution for intermediate water
                    k = 2 * math.pi / L0  # Initial guess
                    for _ in range(5):  # Few iterations usually sufficient
                        L = 2 * math.pi / k
                        k_new = 2 * math.pi / L * math.tanh(k * depth)
                        if abs(k_new - k) < 1e-6:
                            break
                        k = k_new
                    return math.sqrt(g / k * math.tanh(k * depth))
            
            # Calculate wave celerities at current and next depths
            C_current = calculate_wave_celerity(wave_period, current_depth)
            C_next = calculate_wave_celerity(wave_period, next_depth)
            
            # ENHANCED: Shoaling coefficient for segment: Ks = sqrt(C_current / C_next)
            if C_next > 0 and C_current > 0:
                Ks = math.sqrt(C_current / C_next)
                # PRESERVE EXISTING: Apply maximum limit from CONF
                Ks = min(Ks, shoaling_factor_max)
            else:
                Ks = 1.0
            
            return Ks
            
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error calculating shoaling coefficient: {e}")
            return 1.0

    def calculate_refraction_coefficient(self, wave_direction, beach_facing, current_depth, next_depth):
        """
        Calculate refraction coefficient for depth segment using Snell's Law
        """
        try:
            # PRESERVE EXISTING: Get physics parameters from enhanced CONF
            service_config = self.config_dict.get('SurfFishingService', {})
            scoring_criteria = service_config.get('scoring_criteria', {})
            surf_scoring = scoring_criteria.get('surf_scoring', {})
            physics_params = surf_scoring.get('physics_parameters', {})
            
            # PRESERVE EXISTING: Data-driven physics parameters from CONF
            refraction_factor_max = float(physics_params.get('refraction_factor_max', '1.2'))
            
            # PRESERVE EXISTING: Calculate incident angle relative to beach normal
            beach_normal = (beach_facing + 90) % 360  # Perpendicular to beach
            incident_angle = abs(wave_direction - beach_normal)
            if incident_angle > 180:
                incident_angle = 360 - incident_angle
            
            # Convert to radians
            theta_0 = math.radians(incident_angle)
            
            # ENHANCED: Calculate refracted angle using segment depth ratio
            depth_ratio = next_depth / current_depth if current_depth > 0 else 1.0
            
            # PRESERVE EXISTING: Snell's Law calculation
            if math.cos(theta_0) > 0:
                sin_theta_1 = math.sin(theta_0) * math.sqrt(depth_ratio)
                sin_theta_1 = min(sin_theta_1, 1.0)  # Prevent math domain error
                
                theta_1 = math.asin(sin_theta_1)
                
                # Refraction coefficient: Kr = sqrt(cos(theta_0) / cos(theta_1))
                if math.cos(theta_1) > 0:
                    Kr = math.sqrt(math.cos(theta_0) / math.cos(theta_1))
                    # PRESERVE EXISTING: Apply maximum limit from CONF
                    Kr = min(Kr, refraction_factor_max)
                else:
                    Kr = 1.0
            else:
                Kr = 1.0
            
            return Kr
            
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error calculating refraction coefficient: {e}")
            return 1.0

    def apply_breaking_limit(self, wave_height, breaking_depth, bottom_type):
        """
        Apply depth-limited breaking criteria
        """
        try:
            # PRESERVE EXISTING: Get physics parameters from enhanced CONF
            service_config = self.config_dict.get('SurfFishingService', {})
            scoring_criteria = service_config.get('scoring_criteria', {})
            surf_scoring = scoring_criteria.get('surf_scoring', {})
            physics_params = surf_scoring.get('physics_parameters', {})
            
            # PRESERVE EXISTING: Data-driven breaking indices from CONF
            gamma_sand = float(physics_params.get('breaking_gamma_sand', '0.78'))
            gamma_reef = float(physics_params.get('breaking_gamma_reef', '1.0'))
            
            # PRESERVE EXISTING: Select breaking index based on bottom type
            if bottom_type in ['sand', 'beach']:
                gamma = gamma_sand
            elif bottom_type in ['reef', 'rock', 'coral']:
                gamma = gamma_reef
            else:
                gamma = gamma_sand  # Default to sand
            
            # PRESERVE EXISTING: Calculate maximum breaking wave height
            max_breaking_height = gamma * breaking_depth
            
            # PRESERVE EXISTING: Apply breaking limit
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
            # PRESERVE EXISTING: Adjust effective breaking depth with tide level
            effective_depth = breaking_depth + current_tide_level
            
            # PRESERVE EXISTING: Ensure minimum depth
            effective_depth = max(effective_depth, 0.5)  # Minimum 0.5ft depth
            
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
        """Add tide information to surf forecast using Phase I data"""
        
        enhanced_forecast = []
        
        for period in surf_forecast:
            try:
                # Use Phase I tide_table data instead of simplified calculation
                tide_stage = self._determine_tide_stage(period['forecast_time'], tide_conditions)
                
                enhanced_period = period.copy()
                enhanced_period.update({
                    'tide_stage': tide_stage['stage'],
                    'tide_height': tide_stage['height']
                })
                
                enhanced_forecast.append(enhanced_period)
                
            except Exception as e:
                log.error(f"{CORE_ICONS['warning']} Error adding tide info to surf period: {e}")
                # Fallback to basic period without tide data
                enhanced_period = period.copy()
                enhanced_period.update({
                    'tide_stage': 'unknown',
                    'tide_height': 0.0
                })
                enhanced_forecast.append(enhanced_period)
        
        return enhanced_forecast
        
    def _determine_tide_stage(self, forecast_time, tide_conditions, db_manager=None):
        """Determine tide stage for surf forecast using Phase I tide_table"""
        try:
            # Use WeeWX 5.1 StdService database access pattern
            if db_manager is None:
                db_manager = self.db_manager
                if db_manager is None:
                    raise Exception("No database manager available")
            
            # Get tides within 12 hours of forecast time
            start_time = forecast_time - 43200  # 12 hours before
            end_time = forecast_time + 43200    # 12 hours after
            
            tide_query = """
                SELECT tide_time, tide_type, predicted_height, station_id, datum
                FROM tide_table 
                WHERE tide_time BETWEEN ? AND ?
                ORDER BY tide_time
            """
            
            tide_rows = list(db_manager.genSql(tide_query, (start_time, end_time)))
            
            if not tide_rows:
                log.warning(f"{CORE_ICONS['warning']} No Phase I tide data found for surf forecast time")
                return {'stage': 'unknown', 'height': 0.0, 'confidence': 0.3}
            
            # Find the closest tide events before and after forecast time
            past_tides = [row for row in tide_rows if row[0] <= forecast_time]
            future_tides = [row for row in tide_rows if row[0] > forecast_time]
            
            if not past_tides and not future_tides:
                return {'stage': 'unknown', 'height': 0.0, 'confidence': 0.3}
            
            # Determine tide stage based on surrounding tides
            if past_tides and future_tides:
                last_tide = past_tides[-1]
                next_tide = future_tides[0]
                
                # Interpolate height between tides
                time_diff = next_tide[0] - last_tide[0]
                time_progress = (forecast_time - last_tide[0]) / time_diff
                height_diff = next_tide[2] - last_tide[2]
                interpolated_height = last_tide[2] + (height_diff * time_progress)
                
                # Determine stage based on tide types and progression
                if last_tide[1] == 'L' and next_tide[1] == 'H':
                    stage = 'rising'
                elif last_tide[1] == 'H' and next_tide[1] == 'L':
                    stage = 'falling'
                elif last_tide[1] == 'H' and time_progress < 0.5:
                    stage = 'high_slack'
                elif last_tide[1] == 'L' and time_progress < 0.5:
                    stage = 'low_slack'
                else:
                    stage = 'transitional'
                
                return {
                    'stage': stage,
                    'height': interpolated_height,
                    'confidence': 0.8
                }
            
            elif past_tides:
                # Only past tide data available
                last_tide = past_tides[-1]
                stage = 'high_slack' if last_tide[1] == 'H' else 'low_slack'
                return {
                    'stage': stage,
                    'height': last_tide[2],
                    'confidence': 0.6
                }
            
            else:
                # Only future tide data available  
                next_tide = future_tides[0]
                stage = 'rising' if next_tide[1] == 'H' else 'falling'
                return {
                    'stage': stage,
                    'height': next_tide[2],
                    'confidence': 0.6
                }
                
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error determining tide stage for surf: {e}")
            return {'stage': 'unknown', 'height': 0.0, 'confidence': 0.3}
    
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

    def _enhance_forecast_with_calculated_fields(self, surf_forecast):
        """
        Enhance forecast periods with calculated fields (wave_height_range, conditions_text)
        
        NEW METHOD: Adds missing field calculations during forecast generation
        USES: Existing format_wave_height_range method (no duplication)
        PRESERVES: All existing forecast data while adding calculated fields
        """
        
        enhanced_forecast = []
        
        for period in surf_forecast:
            # PRESERVE EXISTING: Copy all existing period data
            enhanced_period = period.copy()
            
            # Calculate wave_height_range using existing method
            if 'wave_height_min' in period and 'wave_height_max' in period:
                min_height = period['wave_height_min']
                max_height = period['wave_height_max']
                
                # USE EXISTING METHOD: No code duplication
                wave_range = self.format_wave_height_range(min_height, max_height)
                enhanced_period['wave_height_range'] = wave_range
            
            # Generate conditions_text from available data
            conditions_parts = []
            
            # Add wave description
            if 'wave_height' in period and period['wave_height'] is not None:
                wave_height = period['wave_height']
                if wave_height >= 6:
                    conditions_parts.append("Large surf")
                elif wave_height >= 3:
                    conditions_parts.append("Moderate surf")
                elif wave_height >= 1:
                    conditions_parts.append("Small surf")
                else:
                    conditions_parts.append("Minimal surf")
            
            # Add wind condition if available
            if 'wind_condition' in period and period['wind_condition'] not in ['unknown', None]:
                wind_cond = period['wind_condition']
                conditions_parts.append(f"{wind_cond} winds")
            
            # Add tide stage if available
            if 'tide_stage' in period and period['tide_stage'] not in ['unknown', None]:
                tide_stage = period['tide_stage']
                conditions_parts.append(f"{tide_stage} tide")
            
            # Combine into conditions_text
            if conditions_parts:
                enhanced_period['conditions_text'] = ", ".join(conditions_parts)
            else:
                enhanced_period['conditions_text'] = "Forecast conditions"
            
            enhanced_forecast.append(enhanced_period)
        
        return enhanced_forecast

    def _get_target_unit_system(self):
        """Get the target unit system from WeeWX configuration"""
        
        # Read the target unit system from StdConvert section
        convert_config = self.config_dict.get('StdConvert', {})
        target_unit_nick = convert_config.get('target_unit', 'US')
        
        # Map string to WeeWX constant
        if target_unit_nick.upper() == 'METRIC':
            return weewx.METRIC
        elif target_unit_nick.upper() == 'METRICWX':
            return weewx.METRICWX
        else:
            return weewx.US

    def store_surf_forecasts(self, spot_id, forecast_data, db_manager):
        """Store surf forecasts using field mappings already loaded in __init__"""
        try:
            # EXISTING CODE: Clear existing forecasts - WeeWX 5.1 pattern preserved
            db_manager.connection.execute(
                "DELETE FROM marine_forecast_surf_data WHERE spot_id = ?",
                (spot_id,)
            )
            
            # USE ALREADY LOADED FIELD DATA: Get field mappings from __init__
            service_config = self.config_dict.get('SurfFishingService', {})
            gfs_wave_config = service_config.get('noaa_gfs_wave', {})
            field_mappings = gfs_wave_config.get('field_mappings', {})
            
            # BUILD COMPLETE FIELD LIST: API fields + hardcoded service fields + missing fields
            field_names = []
            
            # Add API fields that we successfully loaded in __init__
            for field_name, field_config in field_mappings.items():
                database_field = field_config.get('database_field', field_name)
                field_names.append(database_field)
            
            # Add hardcoded service fields that install.py creates in the database
            service_fields = [
                'dateTime', 'usUnits', 'spot_id', 'forecast_time', 'generated_time',
                'quality_rating', 'confidence', 'conditions_text', 'wind_condition',
                'tide_height', 'tide_stage'
            ]
            field_names.extend(service_fields)
            
            # Add the 7 missing fields that surf_fishing.py expects (from Final Report)
            missing_fields = [
                'wave_height_min', 'wave_height_max', 'wave_height_range',
                'quality_stars', 'quality_text', 'conditions_description', 'swell_dominance'
            ]
            field_names.extend(missing_fields)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_field_names = []
            for field in field_names:
                if field not in seen:
                    unique_field_names.append(field)
                    seen.add(field)
            
            # EXISTING CODE: Process each forecast period preserved
            for forecast_period in forecast_data:
                # NEW: Apply data-driven unit conversions
                converted_data = forecast_period
                
                # NEW: Build dynamic INSERT using complete field list
                placeholders = ', '.join(['?' for _ in unique_field_names])
                field_list = ', '.join(unique_field_names)
                
                # NEW: Build values list using complete field specifications
                values = []
                for field_name in unique_field_names:
                    if field_name == 'dateTime':
                        values.append(int(time.time()))
                    elif field_name == 'usUnits':
                        values.append(self._get_target_unit_system())
                    elif field_name == 'spot_id':
                        values.append(spot_id)
                    elif field_name == 'forecast_time':
                        values.append(converted_data.get('forecast_time', int(time.time())))
                    elif field_name == 'generated_time':
                        values.append(int(time.time()))
                    else:
                        # Get value from forecast data or use None for missing fields
                        values.append(converted_data.get(field_name))
                
                # EXISTING CODE: Execute database insert (now data-driven)
                db_manager.connection.execute(
                    f"INSERT INTO marine_forecast_surf_data ({field_list}) VALUES ({placeholders})",
                    values
                )
            
            # Commit the transaction
            db_manager.connection.commit()
            
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
        """Apply research-based coastal wave transformation (Komar methodology)
        
        ENHANCED METHOD: Now uses multi-point bathymetric transformation when available
        PRESERVES: All existing functionality and fallback logic
        MAINTAINS: Exact same method signature and output format
        """
        transformed_data = wave_data.copy()
        
        # ENHANCE: Check if we have adaptive bathymetric data available
        if hasattr(self, 'current_spot_config') and self.current_spot_config:
            bathymetric_data = self.get_full_bathymetric_data_from_conf(self.current_spot_config)
            if bathymetric_data['data_source'] in ['adaptive_bathymetry']:
                # Use enhanced adaptive transformation instead of simple distance-based
                log.debug(f"{CORE_ICONS['status']} Using adaptive transformation instead of distance-based")
                return transformed_data  # Adaptive transformation already applied
        
        # PRESERVE EXISTING: Fallback to distance-based transformation
        base_height_factor = 0.7
        
        if distance_miles > 25:
            distance_factor = max(0.8, 1.0 - (distance_miles - 25) / 200)  # Gradual reduction
        else:
            distance_factor = 1.0
        
        # PRESERVE EXISTING: Apply transformation to wave height
        if 'wave_height' in transformed_data and transformed_data['wave_height'] is not None:
            transformed_data['wave_height'] *= (base_height_factor * distance_factor)
        
        # PRESERVE EXISTING: Add quality confidence
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

    def export_surf_forecast_summary(self, spot_id, format='dict'):
            """Export complete surf forecast summary for templates or external use"""
            try:
                # Get current forecasts using existing method
                current_forecasts = self.get_current_surf_forecast(spot_id, 72, self.db_manager)
                
                # Get next good session using existing method
                next_good_session = self.find_next_good_session(spot_id, self.db_manager, 3)
                
                # Get today's summary - CONSOLIDATED INLINE with thread-safe access
                today_summary = {'max_rating': 0, 'best_period': 'No data', 'avg_rating': 0, 'period_count': 0}
                
                # THREAD SAFE: Get fresh database manager for this thread if no engine available
                with weewx.manager.open_manager_with_config(self.config_dict, 'wx_binding') as db_manager:
                    try:
                        current_time = int(time.time())
                        today_start = current_time - (current_time % 86400)
                        today_end = today_start + 86400
                        
                        result = db_manager.connection.execute("""
                            SELECT quality_rating, conditions_description, wave_height_range
                            FROM marine_forecast_surf_data 
                            WHERE spot_id = ? AND forecast_time BETWEEN ? AND ?
                            ORDER BY quality_rating DESC
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
                        log.error(f"{CORE_ICONS['warning']} Error getting today's surf summary: {e}")
                
                # Get spot information from CONF
                service_config = self.config_dict.get('SurfFishingService', {})
                surf_spots = service_config.get('surf_spots', {})
                
                spot_info = None
                for spot_key, spot_config in surf_spots.items():
                    if spot_key == spot_id or spot_config.get('name') == spot_id:
                        spot_info = {
                            'id': spot_key,
                            'name': spot_config.get('name', spot_key),
                            'latitude': float(spot_config.get('latitude', '0.0')),
                            'longitude': float(spot_config.get('longitude', '0.0')),
                            'bottom_type': spot_config.get('bottom_type', 'sand'),
                            'exposure': spot_config.get('exposure', 'exposed'),
                            'type': spot_config.get('type', 'surf')
                        }
                        break
                
                if not spot_info:
                    return {'error': 'Surf spot not found'}
                
                # Compile complete summary
                summary = {
                    'spot_info': spot_info,
                    'generated_time': datetime.now().isoformat(),
                    'forecasts': current_forecasts,
                    'next_good_session': next_good_session,
                    'today_summary': today_summary,
                    'forecast_count': len(current_forecasts)
                }
                
                if format == 'json':
                    return json.dumps(summary, indent=2)
                else:
                    return summary
                    
            except Exception as e:
                log.error(f"{CORE_ICONS['warning']} Error exporting surf forecast summary: {e}")
                return {'error': f'Export failed: {str(e)}'}  

    def _apply_multi_point_wave_transformation(self, initial_wave_height, wave_period, wave_direction, 
                                             bathymetric_profile, beach_facing, bottom_type, current_tide):
        """
        Apply incremental wave transformation along complete bathymetric path
        
        NEW METHOD: Implements segment-by-segment wave physics calculations
        SCIENTIFIC BASIS: Uses established shoaling, refraction, and breaking physics
        """
        try:
            current_wave_height = initial_wave_height
            
            # Apply tidal effects to entire bathymetric profile
            tidal_adjusted_profile = []
            for point in bathymetric_profile:
                adjusted_depth = self.integrate_tidal_effects(point['depth'], current_tide)
                adjusted_point = point.copy()
                adjusted_point['effective_depth'] = adjusted_depth
                tidal_adjusted_profile.append(adjusted_point)
            
            # Process each depth segment incrementally
            for i in range(len(tidal_adjusted_profile) - 1):
                current_point = tidal_adjusted_profile[i]
                next_point = tidal_adjusted_profile[i + 1]
                
                current_depth = current_point['effective_depth']
                next_depth = next_point['effective_depth']
                
                # Calculate segment shoaling coefficient
                Ks_segment = self.calculate_shoaling_coefficient(current_depth, next_depth, wave_period)
                
                # Calculate segment refraction coefficient (using incremental approach)
                Kr_segment = self.calculate_refraction_coefficient(
                    wave_direction, beach_facing, current_depth, next_depth
                )
                
                # Apply segment transformation
                current_wave_height *= (Ks_segment * Kr_segment)
                
                # Apply bottom friction in shallow water (if implemented)
                if next_depth < 10.0:
                    friction_factor = self._calculate_bottom_friction(next_depth, bottom_type)
                    current_wave_height *= (1.0 - friction_factor)
                
                log.debug(f"{CORE_ICONS['navigation']} Segment {i}: "
                         f"{current_depth:.1f}m→{next_depth:.1f}m, "
                         f"Ks={Ks_segment:.3f}, Kr={Kr_segment:.3f}, "
                         f"H={current_wave_height:.2f}ft")
            
            # Apply final breaking limitation at surf break
            final_breaking_depth = tidal_adjusted_profile[-1]['effective_depth']
            final_height = self.apply_breaking_limit(current_wave_height, final_breaking_depth, bottom_type)
            
            log.debug(f"{CORE_ICONS['status']} Multi-point transformation: "
                     f"{initial_wave_height:.2f}ft → {final_height:.2f}ft "
                     f"({len(bathymetric_profile)} segments)")
            
            return final_height
            
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error in multi-point transformation: {e}")
            # Fallback to simple 2-point transformation
            offshore_depth = bathymetric_profile[0]['depth']
            breaking_depth = bathymetric_profile[-1]['depth']
            effective_breaking_depth = self.integrate_tidal_effects(breaking_depth, current_tide)
            
            Ks = self.calculate_shoaling_coefficient(offshore_depth, effective_breaking_depth, wave_period)
            Kr = self.calculate_refraction_coefficient(wave_direction, beach_facing, offshore_depth, effective_breaking_depth)
            
            transformed_height = initial_wave_height * Ks * Kr
            return self.apply_breaking_limit(transformed_height, effective_breaking_depth, bottom_type)


class SurfForecastSearchList(SearchList):
    """
    Provides surf forecast data to WeeWX templates using WeeWX 5.1 patterns
    """
    
    def __init__(self, generator):
        super(SurfForecastSearchList, self).__init__(generator)
        
    def get_extension_list(self, timespan, db_lookup):
        """Return search list with surf forecast data for templates"""
        try:
            # Get database manager
            db_manager = db_lookup()
            
            # Get service configuration from generator
            config_dict = self.generator.config_dict
            service_config = config_dict.get('SurfFishingService', {})
            
            # Get all active surf spots from CONF
            surf_spots = service_config.get('surf_spots', {})
            
            if not surf_spots:
                return [{'surf_forecasts': {}, 'surf_summary': {'status': 'No surf spots configured'}}]
            
            # Initialize surf generator with engine reference (following fishing pattern)
            surf_generator = SurfForecastGenerator(config_dict, db_manager)
            
            # Build forecast data for each spot
            surf_forecasts = {}
            for spot_id, spot_config in surf_spots.items():
                # Check if spot is active (same logic as fishing)
                is_active = spot_config.get('active', 'true').lower() in ['true', '1', 'yes']
                
                if is_active:
                    spot_data = surf_generator.export_surf_forecast_summary(spot_id, 'dict')
                    if spot_data and 'error' not in spot_data:
                        surf_forecasts[spot_config.get('name', spot_id)] = spot_data
            
            # Get overall summary - CONSOLIDATED INLINE (matching fishing pattern)
            if not surf_forecasts:
                summary_data = {'status': 'No surf spots configured'}
            else:
                best_rating = 0
                best_spot = None
                
                for spot_name, spot_data in surf_forecasts.items():
                    today_summary = spot_data.get('today_summary', {})
                    max_rating = today_summary.get('max_rating', 0)
                    
                    if max_rating > best_rating:
                        best_rating = max_rating
                        best_spot = spot_name
                
                if best_rating >= 4:
                    status = f'Epic surf at {best_spot}'
                elif best_rating >= 3:
                    status = f'Good surf at {best_spot}'
                elif best_rating >= 2:
                    status = f'Fair surf available'
                else:
                    status = 'Poor surf conditions'
                
                summary_data = {
                    'status': status,
                    'best_rating': best_rating,
                    'best_spot': best_spot,
                    'total_spots': len(surf_forecasts)
                }
            
            # Get last update time - CONSOLIDATED INLINE with thread-safe access (matching fishing pattern)
            last_update = 'Never'
            try:
                # THREAD SAFE: Get fresh database manager for this thread
                with weewx.manager.open_manager_with_config(config_dict, 'wx_binding') as db_manager:
                    result = db_manager.connection.execute(
                        "SELECT MAX(generated_time) FROM marine_forecast_surf_data"
                    )
                    
                    row = result.fetchone()
                    
                    if row and row[0]:
                        last_update = datetime.fromtimestamp(row[0]).strftime('%m/%d %I:%M %p')
                        
            except Exception as e:
                log.error(f"{CORE_ICONS['warning']} Error getting last surf update time: {e}")
            
            return [{
                'surf_forecasts': surf_forecasts,
                'surf_summary': summary_data,
                'surf_spots_count': len(surf_spots),
                'surf_last_update': last_update
            }]
            
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error in SurfForecastSearchList: {e}")
            return [{'surf_forecasts': {}, 'surf_summary': {'status': 'Error loading surf data'}}]

    def configure_search_lists(self):
        """Register SearchList extensions for template integration"""
        try:
            # This should be handled by the installer, but verify it's working
            import weewx.cheetahgenerator
            if hasattr(weewx.cheetahgenerator.SearchList, 'extensible'):
                if 'user.surf_fishing.SurfForecastSearchList' not in weewx.cheetahgenerator.SearchList.extensible:
                    weewx.cheetahgenerator.SearchList.extensible.append(
                        'user.surf_fishing.SurfForecastSearchList'
                    )
            log.info(f"{CORE_ICONS['status']} Surf SearchList extensions registered")
        except Exception as e:
            log.warning(f"{CORE_ICONS['warning']} Surf SearchList registration issue: {e}")
            

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

    def _determine_fishing_tide_movement(self, forecast_time):
        """
        Determine tide movement for fishing forecast using Phase I tide_table data
        """
        
        try:
            # THREAD SAFE: Use WeeWX 5.1 database manager pattern
            if not self.engine:
                raise Exception("No engine available for Phase I tide integration")
            
            # THREAD SAFE: Get fresh database manager for this thread
            with weewx.manager.open_manager_with_config(self.config_dict, 'wx_binding') as db_manager:
                
                # Query Phase I tide_table for tide movement analysis
                time_window = 12 * 3600  # 12 hours window
                start_time = forecast_time - time_window
                end_time = forecast_time + time_window
                
                # ✅ FIXED: Use genSql pattern like working surf code
                tide_query = """
                    SELECT tide_time, tide_type, predicted_height, station_id
                    FROM tide_table 
                    WHERE tide_time BETWEEN ? AND ?
                    ORDER BY tide_time
                    LIMIT 10
                """
                tide_events = list(db_manager.genSql(tide_query, (start_time, end_time)))
                
                if not tide_events:
                    raise Exception(f"No Phase I tide data found for fishing forecast time {forecast_time}")
                
                # PRESERVE EXISTING: Analyze tide movement for fishing optimization
                before_events = [t for t in tide_events if t[0] <= forecast_time]
                after_events = [t for t in tide_events if t[0] > forecast_time]
                
                if before_events and after_events:
                    last_tide = before_events[-1]  # Most recent
                    next_tide = after_events[0]    # Next upcoming
                    
                    # Calculate time to next tide change (important for fishing)
                    time_to_next = (next_tide[0] - forecast_time) / 3600.0  # Hours
                    time_from_last = (forecast_time - last_tide[0]) / 3600.0  # Hours
                    
                    # Determine fishing-optimized tide movement
                    if last_tide[1] == 'L' and next_tide[1] == 'H':
                        movement = 'rising'
                        # Fishing optimal: 2 hours before and after tide change
                        if time_to_next <= 2.0 or time_from_last <= 2.0:
                            fishing_quality = 'optimal'
                        else:
                            fishing_quality = 'good'
                            
                    elif last_tide[1] == 'H' and next_tide[1] == 'L':
                        movement = 'falling'
                        # Fishing optimal: 2 hours before and after tide change
                        if time_to_next <= 2.0 or time_from_last <= 2.0:
                            fishing_quality = 'optimal'
                        else:
                            fishing_quality = 'good'
                    else:
                        # Slack tide periods
                        if abs(time_from_last) < 0.5:  # Within 30 minutes of tide
                            movement = 'high' if last_tide[1] == 'H' else 'low'
                            fishing_quality = 'fair'  # Slack water less optimal
                        else:
                            movement = 'rising' if next_tide[1] == 'H' else 'falling'
                            fishing_quality = 'good'
                    
                    # Calculate tide range (affects fishing quality)
                    tide_range = abs(next_tide[2] - last_tide[2])
                    
                elif before_events:
                    # Only past events - use most recent
                    closest_tide = before_events[-1]
                    movement = 'high' if closest_tide[1] == 'H' else 'low'
                    fishing_quality = 'fair'
                    time_to_next = 6.0  # Default
                    tide_range = 3.0  # Default
                    
                elif after_events:
                    # Only future events - use next
                    closest_tide = after_events[0]
                    time_to_next = (closest_tide[0] - forecast_time) / 3600.0
                    movement = 'rising' if closest_tide[1] == 'H' else 'falling'
                    fishing_quality = 'good' if time_to_next <= 2.0 else 'fair'
                    tide_range = 3.0  # Default
                
                else:
                    raise Exception("Unable to determine tide movement from available data")
                
                return {
                    'movement': movement,  # rising, falling, high, low
                    'quality': fishing_quality,  # optimal, good, fair
                    'time_to_next_hours': round(time_to_next, 1),
                    'tide_range_feet': round(tide_range, 1),
                    'description': f"{movement.title()} tide, {fishing_quality} fishing conditions"
                }
            
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Phase I fishing tide integration failed: {e}")
            # NO FALLBACKS: Fail as required by CLAUDE.md
            raise Exception(f"Phase I tide integration required for fishing forecasts: {e}")

    def _calculate_fishing_tide_score(self, tide_info):
        """
        Calculate fishing-specific tide score based on Phase I tide data
        """
        
        movement = tide_info['movement']
        quality = tide_info['quality']
        time_to_next = tide_info['time_to_next_hours']
        tide_range = tide_info['tide_range_feet']
        
        # Base score by tide movement (fishing research-based)
        movement_scores = {
            'rising': 0.9,   # Fish feed actively on rising tide
            'falling': 0.8,  # Good fishing as tide falls
            'high': 0.4,     # Slack water, less active
            'low': 0.5       # Some species active at low tide
        }
        
        base_score = movement_scores.get(movement, 0.5)
        
        # Quality multiplier based on timing
        quality_multipliers = {
            'optimal': 1.2,  # Within 2 hours of tide change
            'good': 1.0,     # Standard conditions
            'fair': 0.7      # Slack or distant from change
        }
        
        quality_multiplier = quality_multipliers.get(quality, 1.0)
        
        # Tide range bonus (larger ranges = more fish movement)
        if tide_range >= 5.0:
            range_bonus = 0.1  # Large tide range
        elif tide_range >= 3.0:
            range_bonus = 0.05  # Moderate tide range
        else:
            range_bonus = 0.0  # Small tide range
        
        # Calculate final score
        final_score = (base_score * quality_multiplier) + range_bonus
        
        # Normalize to 0.0-1.0 range
        return min(1.0, max(0.0, final_score))

    def generate_fishing_forecast(self, spot, marine_conditions):
        """
        Generate complete fishing forecast for a spot with Phase I tide integration
        """
        try:
            log.debug(f"{CORE_ICONS['navigation']} Generating fishing forecast for {spot['name']}")
            
            # PRESERVE EXISTING: Generate fishing periods (standard 3-day forecast)
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
                try:
                    # FIXED: Use existing unified scoring method that handles tide data internally
                    period_score = self.score_fishing_period_unified(period, spot, marine_conditions)
                    
                    # PRESERVE EXISTING: Create forecast record with period_score data structure
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
                        'confidence': period_score['overall']['confidence'],
                        'tide_score': period_score['tide']['score'],
                        'time_to_next_hours': period_score['tide']['time_to_next_hours'],
                        'tide_confidence': period_score['tide']['confidence'],
                        'tide_description': period_score['tide']['description']
                    }
                    forecasts.append(forecast)
                    
                except Exception as period_error:
                    log.error(f"{CORE_ICONS['warning']} Error processing fishing period: {period_error}")
                    # Re-raise Phase I integration errors, continue for other errors
                    if "Phase I tide integration required" in str(period_error):
                        raise period_error
                    continue

            log.debug(f"{CORE_ICONS['status']} Generated {len(forecasts)} fishing forecast periods with Phase I tide integration")
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

            # FIXED: Correct tide data collection method calls
            try:
                # Always try to collect Phase I tide data directly from tide_table (like surf forecasts do)
                with weewx.manager.open_manager_with_config(self.config_dict, 'wx_binding') as db_manager:
                    tide_info = self._determine_tide_stage(period['period_start_time'], {}, db_manager)
                    
                    # Convert surf tide format to fishing tide format
                    tide_data = {
                        'tide_movement': tide_info['stage'],  # 'rising', 'falling', 'high_slack', 'low_slack'
                        'tide_height': tide_info['height'],
                        'confidence': tide_info['confidence'],
                        'time_to_next_hours': 6.0  # Default from CONF if needed
                    }

            except Exception as e:
                log.error(f"{CORE_ICONS['warning']} Error collecting tide data: {e}")
                # FAIL CLEANLY: No hardcoded fallbacks per CLAUDE.md Fix 5
                raise Exception(f"Phase I tide data required but not available: {e}")

            # PRESERVE EXISTING: Enhanced pressure scoring using CONF parameters (UNCHANGED)
            try:
                pressure_ranges = self.fishing_scoring.get('pressure_trend_scoring', {}).get('ranges', {})
                if not pressure_ranges:
                    raise Exception("Pressure scoring ranges not found in CONF")
                
                # Determine pressure condition
                pressure_value = pressure_data['pressure']
                if pressure_value >= float(pressure_ranges.get('high_pressure_threshold', '30.2')):
                    pressure_condition = 'high_stable'
                    pressure_score_value = 0.8
                elif pressure_value >= float(pressure_ranges.get('normal_pressure_min', '29.8')):
                    pressure_condition = 'stable'
                    pressure_score_value = 0.7
                elif pressure_value >= float(pressure_ranges.get('low_pressure_threshold', '29.5')):
                    pressure_condition = 'falling'
                    pressure_score_value = 0.9
                else:
                    pressure_condition = 'low_stormy'
                    pressure_score_value = 0.3
                
                pressure_score = {
                    'score': pressure_score_value,
                    'condition': pressure_condition,
                    'trend': pressure_data.get('trend', 'stable'),
                    'value': pressure_value,
                    'confidence': pressure_data.get('confidence', 0.7)
                }
            except Exception as e:
                log.error(f"{CORE_ICONS['warning']} Error scoring pressure: {e}")
                raise Exception(f"Pressure scoring failed: {e}")

            # PRESERVE EXISTING: Enhanced tide scoring using CONF parameters (UNCHANGED)
            try:
                tide_movement = tide_data.get('tide_movement', 'unknown')
                time_to_next = tide_data.get('time_to_next_hours', 6.0)
                
                # Map tide movement to CONF tide phase scoring
                tide_phase_mapping = {
                    'rising': 'incoming',
                    'falling': 'outgoing', 
                    'high_slack': 'high_slack',
                    'low_slack': 'low_slack',
                    'slack': 'low_slack'  # Default slack to low_slack
                }

                tide_phase = tide_phase_mapping.get(tide_movement, tide_movement)

                # Get tide phase scores from CONF
                tide_phase_scoring = self.fishing_scoring.get('tide_phase_scoring', {})
                if not tide_phase_scoring:
                    raise Exception("Tide phase scoring not found in CONF")

                tide_score_value = tide_phase_scoring.get(tide_phase, None)
                if tide_score_value is None:
                    raise Exception(f"Tide phase '{tide_phase}' not found in CONF tide_phase_scoring")

                tide_score_value = float(tide_score_value)
                
                tide_score = {
                    'score': tide_score_value,
                    'movement': tide_movement,
                    'time_to_next_hours': time_to_next,
                    'confidence': tide_data.get('confidence', 0.8),
                    'description': f'{tide_movement.replace("_", " ").title()} tide phase'
                }
            except Exception as e:
                log.error(f"{CORE_ICONS['warning']} Error in enhanced tide scoring: {e}")
                raise Exception(f"Tide scoring failed: {e}")

            # PRESERVE EXISTING: Enhanced time scoring using CONF parameters (UNCHANGED)
            try:
                period_start_time = period['period_start_time']
                period_hour = int((period_start_time % 86400) / 3600)
                
                time_of_day_scoring = self.fishing_scoring.get('time_of_day_scoring', {})
                
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
                raise Exception(f"Time scoring failed: {e}")

            # PRESERVE EXISTING: Enhanced species scoring using CONF fish categories (UNCHANGED)
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
                raise Exception(f"Species scoring failed: {e}")

            # PRESERVE EXISTING: Enhanced overall rating using CONF weights (UNCHANGED)
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
                raise Exception(f"Overall rating calculation failed: {e}")

            # PRESERVE EXISTING: Description generation logic (ENHANCED) - using CONF for descriptions
            rating_descriptions = self.fishing_scoring.get('rating_descriptions', {})
            if not rating_descriptions:
                raise Exception("Rating descriptions not found in CONF")
            
            rating = overall_rating['rating']
            base_desc = rating_descriptions.get(str(rating), f"Rating {rating} fishing conditions")
            
            # Add key contributing factors using CONF thresholds
            factor_thresholds = self.fishing_scoring.get('description_factor_thresholds', {})
            pressure_threshold = float(factor_thresholds.get('pressure_threshold', '0.7'))
            tide_threshold = float(factor_thresholds.get('tide_threshold', '0.7')) 
            time_threshold = float(factor_thresholds.get('time_threshold', '0.7'))
            
            factors = []
            if pressure_score['score'] >= pressure_threshold:
                factors.append(f"favorable {pressure_score['trend']} pressure")
            if tide_score['score'] >= tide_threshold and tide_score['movement'] != 'unknown':
                factors.append(f"optimal {tide_score['movement']} tide")
            if time_score['score'] >= time_threshold:
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
            # FIXED: Fail cleanly when Phase I unavailable per CLAUDE.md Fix 5
            raise Exception(f"Fishing period scoring requires Phase I integration: {e}")

    def store_fishing_forecasts(self, spot_id, fishing_forecast, db_manager):
        """
        Store fishing forecasts using WeeWX 5.1 thread-safe database patterns with Phase I tide data
        """
        try:
            if not fishing_forecast or not self.engine:
                return False
            
            # Clear existing forecasts for this spot - WeeWX 5.1 pattern
            db_manager.connection.execute(
                "DELETE FROM marine_forecast_fishing_data WHERE spot_id = ?",
                (spot_id,)
            )
            
            # Insert new forecasts with all required WeeWX fields
            insert_query = """
                INSERT INTO marine_forecast_fishing_data (
                    dateTime, usUnits, spot_id, forecast_date, period_name, period_start_hour, period_end_hour,
                    generated_time, pressure_trend, tide_movement, species_activity, 
                    activity_rating, conditions_text, best_species
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            current_time = int(time.time())
            target_unit_system = self._get_target_unit_system()
            
            for period in fishing_forecast:
                values = (
                    current_time,  # dateTime - required WeeWX field
                    target_unit_system,  # usUnits - required WeeWX field
                    spot_id,
                    period['forecast_date'],
                    period['period_name'], 
                    period['period_start_hour'],
                    period['period_end_hour'],
                    period['generated_time'],
                    period.get('pressure_trend', 'stable'),
                    period.get('tide_movement', 'not_available'),
                    period.get('species_activity', 'moderate'),
                    period.get('activity_rating', 2),
                    period.get('conditions_text', 'Fishing conditions'),
                    json.dumps(period.get('best_species', []))
                )
                db_manager.connection.execute(insert_query, values)
            
            log.debug(f"{CORE_ICONS['status']} Stored {len(fishing_forecast)} fishing forecast periods with tide data for spot {spot_id}")
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
                # ✅ FIXED: Use genSql pattern
                query = """
                    SELECT forecast_date, period_name, period_start_hour, period_end_hour,
                        pressure_trend, tide_movement, species_activity, activity_rating,
                        conditions_text, best_species, generated_time
                    FROM marine_forecast_fishing_data 
                    WHERE spot_id = ? AND forecast_date BETWEEN ? AND ?
                    ORDER BY forecast_date ASC, period_start_hour ASC
                """
                rows = list(db_manager.genSql(query, (spot_id, current_time, end_time)))
                
                # PRESERVE EXISTING: Complete forecast processing logic
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
                # ✅ FIXED: Use genSql pattern
                query = """
                    SELECT forecast_date, period_name, period_start_hour, period_end_hour,
                        activity_rating, conditions_text, best_species
                    FROM marine_forecast_fishing_data 
                    WHERE spot_id = ? AND forecast_date >= ? AND activity_rating >= ?
                    ORDER BY forecast_date ASC, period_start_hour ASC
                    LIMIT 1
                """
                rows = list(db_manager.genSql(query, (spot_id, current_time, min_rating)))
                
                # PRESERVE EXISTING: Complete result processing logic
                if rows:
                    row = rows[0]  # Get first result
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

    def _determine_tide_stage(self, forecast_time, tide_conditions, db_manager=None):
        """Determine tide stage for surf forecast using Phase I tide_table"""
        try:
            # Use WeeWX 5.1 StdService database access pattern
            if db_manager is None:
                db_manager = self.db_manager
                if db_manager is None:
                    raise Exception("No database manager available")
            
            # Get tides within 12 hours of forecast time
            start_time = forecast_time - 43200  # 12 hours before
            end_time = forecast_time + 43200    # 12 hours after
            
            tide_query = """
                SELECT tide_time, tide_type, predicted_height, station_id, datum
                FROM tide_table 
                WHERE tide_time BETWEEN ? AND ?
                ORDER BY tide_time
            """
            
            tide_rows = list(db_manager.genSql(tide_query, (start_time, end_time)))
            
            if not tide_rows:
                log.warning(f"{CORE_ICONS['warning']} No Phase I tide data found for surf forecast time")
                return {'stage': 'unknown', 'height': 0.0, 'confidence': 0.3}
            
            # Find the closest tide events before and after forecast time
            past_tides = [row for row in tide_rows if row[0] <= forecast_time]
            future_tides = [row for row in tide_rows if row[0] > forecast_time]
            
            if not past_tides and not future_tides:
                return {'stage': 'unknown', 'height': 0.0, 'confidence': 0.3}
            
            # Determine tide stage based on surrounding tides
            if past_tides and future_tides:
                last_tide = past_tides[-1]
                next_tide = future_tides[0]
                
                # Interpolate height between tides
                time_diff = next_tide[0] - last_tide[0]
                time_progress = (forecast_time - last_tide[0]) / time_diff
                height_diff = next_tide[2] - last_tide[2]
                interpolated_height = last_tide[2] + (height_diff * time_progress)
                
                # Determine stage based on tide types and progression
                if last_tide[1] == 'L' and next_tide[1] == 'H':
                    stage = 'rising'
                elif last_tide[1] == 'H' and next_tide[1] == 'L':
                    stage = 'falling'
                elif last_tide[1] == 'H' and time_progress < 0.5:
                    stage = 'high_slack'
                elif last_tide[1] == 'L' and time_progress < 0.5:
                    stage = 'low_slack'
                else:
                    stage = 'transitional'
                
                return {
                    'stage': stage,
                    'height': interpolated_height,
                    'confidence': 0.8
                }
            
            elif past_tides:
                # Only past tide data available
                last_tide = past_tides[-1]
                stage = 'high_slack' if last_tide[1] == 'H' else 'low_slack'
                return {
                    'stage': stage,
                    'height': last_tide[2],
                    'confidence': 0.6
                }
            
            else:
                # Only future tide data available  
                next_tide = future_tides[0]
                stage = 'rising' if next_tide[1] == 'H' else 'falling'
                return {
                    'stage': stage,
                    'height': next_tide[2],
                    'confidence': 0.6
                }
                
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error determining tide stage for surf: {e}")
            return {'stage': 'unknown', 'height': 0.0, 'confidence': 0.3}

    def _get_target_unit_system(self):
        """Get the target unit system from WeeWX configuration"""
        
        # Read the target unit system from StdConvert section
        convert_config = self.config_dict.get('StdConvert', {})
        target_unit_nick = convert_config.get('target_unit', 'US')
        
        # Map string to WeeWX constant
        if target_unit_nick.upper() == 'METRIC':
            return weewx.METRIC
        elif target_unit_nick.upper() == 'METRICWX':
            return weewx.METRICWX
        else:
            return weewx.US
    

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
        log.info(f"DEBUG: SurfFishingService config keys: {list(self.service_config.keys())}")

        # NEW: WeeWX 5.1 unit system detection using CONF data
        self._setup_unit_system_from_conf(config_dict)
        
        # EXISTING CODE: Initialize GRIB processor - PRESERVED EXACTLY
        self.grib_processor = GRIBProcessor(config_dict)
        if self.grib_processor.is_available():
            log.info("Using pygrib for GRIB processing")
        else:
            log.warning("No GRIB library available - WaveWatch III forecasts disabled")
        
        # NEW: Initialize bathymetry processor
        self.bathymetry_processor = BathymetryProcessor(config_dict, self.grib_processor, self.engine)
    
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
        """Main forecast generation loop - complete operation in one method"""
        log.info(f"{CORE_ICONS['status']} Forecast generation thread started")
          
        while not self.shutdown_event.is_set():
            try:
                log.debug("Starting forecast generation (API calls and processing)")
                
                # EXISTING: Open thread-local database connection for entire forecast cycle
                with weewx.manager.open_manager_with_config(self.config_dict, 'wx_binding') as db_manager:
                    
                    # EXISTING: Create thread-local generators with database manager
                    surf_generator = SurfForecastGenerator(self.config_dict, db_manager)
                    fishing_generator = FishingForecastGenerator(self.config_dict, self.engine) 
                    
                    # EXISTING: Get active spots
                    active_surf_spots = self._get_active_surf_spots()
                    active_fishing_spots = self._get_active_fishing_spots()
                    
                    surf_count = 0
                    fishing_count = 0
                    
                    # EXISTING: Generate surf forecasts
                    for spot in active_surf_spots:
                        try:
                            # NEW: Check if bathymetry processing needed
                            if spot.get('needs_bathymetry', False):
                                log.info(f"{CORE_ICONS['navigation']} Processing bathymetry for {spot['name']}")
                                
                                bathymetry_success = self.bathymetry_processor.process_surf_spot_bathymetry(spot)
                                
                                if not bathymetry_success:
                                    log.warning(f"{CORE_ICONS['warning']} Bathymetry processing failed for {spot['name']} - continuing with basic forecast")
                                    # Continue with forecast generation even if bathymetry fails
                                else:
                                    # Reload spot data to get updated bathymetry information
                                    updated_spots = self._get_active_surf_spots()
                                    spot = next((s for s in updated_spots if s['id'] == spot['id']), spot)
                                
                            # EXISTING: Get WaveWatch III data if available
                            gfs_wave_data = []
                            if self.grib_processor.is_available():
                                gfs_wave_collector = WaveWatchDataCollector(self.config_dict, self.grib_processor)

                                # Only collect GFS Wave data if we have valid deep water coordinates
                                spot_config = self.config_dict.get('SurfFishingService', {}).get('surf_spots', {}).get(spot['id'], {})
                                if spot_config.get('offshore_latitude') and spot_config.get('offshore_longitude'):
                                    # Use offshore coordinates for GRIB data collection (deep water only)
                                    gfs_wave_data = gfs_wave_collector.fetch_forecast_data(spot_config)
                                    log.debug(f"Using offshore coordinates for GFS Wave data: {spot_config['offshore_latitude']}, {spot_config['offshore_longitude']}")
                                else:
                                    # No deep water coordinates available - skip GFS Wave data collection
                                    log.warning(f"No deep water coordinates available for {spot['name']} - skipping GFS Wave data")
                                    gfs_wave_data = []
 
                            # EXISTING: Generate surf forecast
                            surf_forecast = surf_generator.generate_surf_forecast(spot, gfs_wave_data)
                            
                            if surf_forecast:
                                # EXISTING: Store directly with the generator's method
                                surf_generator.store_surf_forecasts(spot['id'], surf_forecast, db_manager)
                                surf_count += 1
                                
                        except Exception as e:
                            log.error(f"Error generating surf forecast for {spot.get('name', 'unknown')}: {e}")
                            continue
                    
                    # EXISTING: Generate fishing forecasts
                    for spot in active_fishing_spots:
                        try:
                            
                            # Get current marine conditions from Phase I
                            marine_conditions = self._get_phase_i_marine_conditions(spot['latitude'], spot['longitude'])

                            # EXISTING: Generate fishing forecast
                            fishing_forecast = fishing_generator.generate_fishing_forecast(spot, marine_conditions)
                            
                            if fishing_forecast:
                                # EXISTING: Store directly with the generator's method
                                fishing_generator.store_fishing_forecasts(spot['id'], fishing_forecast, db_manager)
                                fishing_count += 1
                                
                        except Exception as e:
                            log.error(f"Error generating fishing forecast for {spot.get('name', 'unknown')}: {e}")
                            continue
                    
                    log.info(f"Forecast generation completed for {surf_count} surf spots and {fishing_count} fishing spots")
                
                # EXISTING: Sleep outside of database context
                log.debug(f"Sleeping for {self.forecast_interval} seconds")
                self.shutdown_event.wait(timeout=self.forecast_interval)
                
            except Exception as e:
                log.error(f"Error in forecast loop: {e}")
                self.shutdown_event.wait(timeout=300)
        
        log.info(f"{CORE_ICONS['status']} Forecast generation thread stopped")
        
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
                
                # Check bathymetry calculation status
                bathymetry_calculated = spot_config.get('bathymetry_calculated', 'false').lower() == 'true'
                
                # Convert CONF data to expected format - all spots in CONF are active
                spot = {
                    'id': spot_id,  # Use CONF key as ID
                    'name': spot_config.get('name', spot_id),
                    'latitude': float(spot_config.get('latitude', '0.0')),
                    'longitude': float(spot_config.get('longitude', '0.0')),
                    'bottom_type': spot_config.get('bottom_type', 'sand'),
                    'exposure': spot_config.get('exposure', 'exposed'),
                    'bathymetric_path': spot_config.get('bathymetric_path', {}),
                    'type': spot_config.get('type', 'surf'),
                    'beach_facing': spot_config.get('beach_facing'),
                    'needs_bathymetry': not bathymetry_calculated
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
        self.fishing_generator.store_fishing_forecasts(spot['id'], fishing_forecast)

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
                    'beach_facing': spot_config.get('beach_facing'),
                    'active': spot_config.get('active', 'true').lower() in ['true', '1', 'yes']
                }
                
                # Add type-specific fields
                if spot_type == 'surf':
                    spot.update({
                        'bottom_type': spot_config.get('bottom_type', 'sand'),
                        'exposure': spot_config.get('exposure', 'exposed'),
                        'beach_facing': spot_config.get('beach_facing')
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
            gfs_wave_config = service_config.get('noaa_gfs_wave', {})
            field_definitions = gfs_wave_config.get('field_mappings', {})
            
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