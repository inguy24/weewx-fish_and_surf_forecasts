#!/usr/bin/env python3
# Magic Animal: Asshole_AI
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
        self.config_dict = config_dict
        self.grib_library = self._detect_grib_library()
        
    def _detect_grib_library(self):
        """Detect available GRIB processing library"""
        
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
        
        log.warning("No GRIB library available - WaveWatch III forecasts disabled")
        return None
    
    def is_available(self):
        """Check if GRIB processing is available"""
        return self.grib_library is not None
    
    def process_wavewatch_file(self, grib_file_path, target_lat, target_lon):
        """Extract WaveWatch III data for specific location from GRIB file"""
        
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
        """Process GRIB file using eccodes-python"""
        
        import eccodes as ec
        
        data_points = []
        
        try:
            with open(grib_file_path, 'rb') as f:
                while True:
                    msg = ec.codes_grib_new_from_file(f)
                    if msg is None:
                        break
                    
                    try:
                        # Get message metadata
                        param_name = ec.codes_get(msg, 'paramId')
                        forecast_time = ec.codes_get(msg, 'validityTime')
                        
                        # Get nearest grid point
                        lat_val = ec.codes_grib_find_nearest(msg, target_lat, target_lon)[0]['lat']
                        lon_val = ec.codes_grib_find_nearest(msg, target_lat, target_lon)[0]['lon']
                        value = ec.codes_grib_find_nearest(msg, target_lat, target_lon)[0]['value']
                        
                        # Map parameter IDs to our field names
                        param_mapping = {
                            100: 'wave_height',      # Significant wave height
                            103: 'wave_period',      # Primary wave period
                            104: 'wave_direction',   # Primary wave direction
                            165: 'wind_speed_10m',   # 10m wind speed
                            166: 'wind_direction_10m' # 10m wind direction
                        }
                        
                        if param_name in param_mapping:
                            data_points.append({
                                'parameter': param_mapping[param_name],
                                'value': value,
                                'forecast_time': forecast_time,
                                'lat': lat_val,
                                'lon': lon_val
                            })
                    
                    finally:
                        ec.codes_release(msg)
        
        except Exception as e:
            log.error(f"eccodes processing error: {e}")
        
        return data_points
    
    def _process_with_pygrib(self, grib_file_path, target_lat, target_lon):
        """Process GRIB file using pygrib"""
        
        import pygrib
        
        data_points = []
        
        try:
            grbs = pygrib.open(grib_file_path)
            
            # Define parameter mappings for pygrib
            param_names = [
                'Significant height of combined wind waves and swell',
                'Primary wave mean period',
                'Primary wave direction',
                '10 metre U wind component',
                '10 metre V wind component'
            ]
            
            for grb in grbs:
                if grb.name in param_names:
                    # Find nearest grid point
                    data, lats, lons = grb.data()
                    
                    # Find closest grid point
                    lat_idx, lon_idx = self._find_nearest_grid_point(
                        target_lat, target_lon, lats, lons
                    )
                    
                    value = data[lat_idx, lon_idx]
                    
                    # Map to our field names
                    field_mapping = {
                        'Significant height of combined wind waves and swell': 'wave_height',
                        'Primary wave mean period': 'wave_period',
                        'Primary wave direction': 'wave_direction',
                        '10 metre U wind component': 'wind_u_10m',
                        '10 metre V wind component': 'wind_v_10m'
                    }
                    
                    field_name = field_mapping.get(grb.name)
                    if field_name:
                        data_points.append({
                            'parameter': field_name,
                            'value': value,
                            'forecast_time': grb.validityTime,
                            'lat': lats[lat_idx, lon_idx],
                            'lon': lons[lat_idx, lon_idx]
                        })
            
            grbs.close()
        
        except Exception as e:
            log.error(f"pygrib processing error: {e}")
        
        return data_points
    
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
        max_age_hours = self.quality_control.get('max_data_age_hours', 6)
        max_age_seconds = max_age_hours * 3600
        
        data_time = source_data.get('observation_time')
        if not data_time:
            return False  # No timestamp = not fresh
            
        age_seconds = current_time - data_time
        return age_seconds <= max_age_seconds
    

class WaveWatchDataCollector:
    """Collect WaveWatch III offshore wave forecast data"""
    
    def __init__(self, config_dict, grib_processor):
        self.config_dict = config_dict
        self.grib_processor = grib_processor
        
        # Get WaveWatch III configuration
        service_config = config_dict.get('SurfFishingService', {})
        self.wavewatch_config = service_config.get('wavewatch_endpoints', {})
        self.base_url = self.wavewatch_config.get('base_url', '')
        
    def fetch_forecast_data(self, latitude, longitude):
        """Fetch WaveWatch III forecast data for location"""
        
        if not self.grib_processor.is_available():
            log.warning("GRIB processing not available - skipping WaveWatch III data")
            return []
        
        try:
            # Select appropriate grid
            grid_name = self._select_grid(latitude, longitude)
            log.debug(f"Using WaveWatch III grid: {grid_name} for location {latitude}, {longitude}")
            
            # Download GRIB files
            grib_files = self._download_grib_files(grid_name)
            
            # Process GRIB files
            forecast_data = []
            for grib_file in grib_files:
                try:
                    data_points = self.grib_processor.process_wavewatch_file(
                        grib_file, latitude, longitude
                    )
                    forecast_data.extend(data_points)
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(grib_file)
                    except OSError:
                        pass
            
            return self._organize_forecast_data(forecast_data)
        
        except Exception as e:
            log.error(f"Error fetching WaveWatch III data: {e}")
            return []
    
    def _select_grid(self, latitude, longitude):
        """Select appropriate WaveWatch III grid based on location"""
        
        grids = self.wavewatch_config.get('grids', {})
        
        # Check regional grids first (higher resolution)
        for grid_name, grid_config in grids.items():
            if grid_name == 'glo_30m':  # Skip global grid for now
                continue
                
            bounds = grid_config.get('bounds', [])
            if len(bounds) == 4:
                lat_min, lat_max, lon_min, lon_max = bounds
                if lat_min <= latitude <= lat_max and lon_min <= longitude <= lon_max:
                    return grid_name
        
        # Fallback to global grid
        return 'glo_30m'
    
    def _download_grib_files(self, grid_name):
        """Download latest WaveWatch III GRIB files"""
        
        grib_files = []
        
        # Get latest model run (typically 00, 06, 12, 18 UTC)
        current_time = time.gmtime()
        model_hours = [0, 6, 12, 18]
        
        # Find most recent model run
        current_hour = current_time.tm_hour
        latest_run = max([h for h in model_hours if h <= current_hour], default=model_hours[-1])
        
        # If no run today yet, use last run from yesterday
        if current_hour < min(model_hours):
            run_date = time.gmtime(time.time() - 86400)  # Yesterday
            latest_run = model_hours[-1]
        else:
            run_date = current_time
        
        date_str = time.strftime('%Y%m%d', run_date)
        run_str = f"{latest_run:02d}"
        
        # Download first few forecast hours (0, 3, 6, 9, 12 hours)
        forecast_hours = [0, 3, 6, 9, 12]
        
        for fhr in forecast_hours:
            try:
                # Construct GRIB file URL
                filename = f"multi_1.{grid_name}.t{run_str}z.f{fhr:03d}.grib2"
                url = f"{self.base_url}wave{date_str}/{filename}"
                
                # Download to temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.grib2')
                temp_file.close()
                
                urllib.request.urlretrieve(url, temp_file.name)
                grib_files.append(temp_file.name)
                
                log.debug(f"Downloaded WaveWatch III file: {filename}")
                
            except Exception as e:
                log.warning(f"Could not download {filename}: {e}")
                continue
        
        return grib_files
    
    def _organize_forecast_data(self, data_points):
        """Organize raw GRIB data into forecast periods"""
        
        # Group data by forecast time
        forecast_periods = {}
        
        for point in data_points:
            forecast_time = point['forecast_time']
            
            if forecast_time not in forecast_periods:
                forecast_periods[forecast_time] = {}
            
            forecast_periods[forecast_time][point['parameter']] = point['value']
        
        # Convert to list format with unit conversions
        organized_data = []
        
        for forecast_time, parameters in forecast_periods.items():
            # Apply unit conversions
            converted_data = {
                'forecast_time': forecast_time,
                'wave_height': parameters.get('wave_height', 0) * 3.28084,  # m to ft
                'wave_period': parameters.get('wave_period', 0),  # seconds
                'wave_direction': parameters.get('wave_direction', 0),  # degrees
                'wind_speed': self._calculate_wind_speed(parameters),  # m/s to mph
                'wind_direction': self._calculate_wind_direction(parameters)  # degrees
            }
            
            organized_data.append(converted_data)
        
        # Sort by forecast time
        organized_data.sort(key=lambda x: x['forecast_time'])
        
        return organized_data
    
    def _calculate_wind_speed(self, parameters):
        """Calculate wind speed from U/V components or direct value"""
        
        # Check for direct wind speed first
        if 'wind_speed_10m' in parameters:
            return parameters['wind_speed_10m'] * 2.23694  # m/s to mph
        
        # Calculate from U/V components
        u_wind = parameters.get('wind_u_10m', 0)
        v_wind = parameters.get('wind_v_10m', 0)
        
        wind_speed_ms = math.sqrt(u_wind**2 + v_wind**2)
        return wind_speed_ms * 2.23694  # m/s to mph
    
    def _calculate_wind_direction(self, parameters):
        """Calculate wind direction from U/V components or direct value"""
        
        # Check for direct wind direction first
        if 'wind_direction_10m' in parameters:
            return parameters['wind_direction_10m']
        
        # Calculate from U/V components
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
        self.config_dict = config_dict
        self.db_manager = db_manager
        service_config = config_dict.get('SurfFishingService', {})
        self.surf_rating_factors = service_config.get('surf_rating_factors', {})
    
    def generate_surf_forecast(self, spot_config, wavewatch_data=None):
        """Generate surf forecast for a specific spot"""
        
        marine_conditions = self.integrate_optimal_data_sources(spot_config)

        try:
            # Get current conditions from Phase I data
            current_waves = self._get_current_wave_conditions(marine_conditions)
            current_wind = self._get_current_wind_conditions(marine_conditions)
            current_tides = self._get_current_tide_conditions(marine_conditions)
            
            # Combine with WaveWatch III offshore forecast
            offshore_forecast = self._process_offshore_forecast(wavewatch_data)
            
            # Transform offshore to local surf conditions
            local_surf_forecast = self._transform_to_local_conditions(
                offshore_forecast, spot_config, current_waves
            )
            
            # Apply wind quality assessment
            surf_forecast = self._assess_surf_quality(
                local_surf_forecast, current_wind, spot_config
            )
            
            # Add tide information
            surf_forecast = self._add_tide_information(surf_forecast, current_tides)
            
            # Calculate confidence ratings
            surf_forecast = self._calculate_confidence_ratings(surf_forecast)
            
            return surf_forecast
        
        except Exception as e:
            log.error(f"Error generating surf forecast for {spot_config.get('name', 'unknown')}: {e}")
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
    
    def _transform_to_local_conditions(self, offshore_forecast, spot_config, current_waves):
        """Transform offshore wave conditions to local surf conditions"""
        
        transformed_forecast = []
        
        for period in offshore_forecast:
            # Simple transformation algorithm (can be enhanced)
            # Factors: bottom type, exposure, depth, local geography
            
            offshore_height = period['offshore_wave_height']
            offshore_period = period['offshore_wave_period']
            
            # Apply exposure factor
            exposure = spot_config.get('exposure', 'exposed')
            exposure_factors = {
                'exposed': 1.0,      # No reduction
                'semi_protected': 0.7,  # 30% reduction
                'protected': 0.4     # 60% reduction
            }
            exposure_factor = exposure_factors.get(exposure, 1.0)
            
            # Apply bottom type factor for wave height
            bottom_type = spot_config.get('bottom_type', 'sand')
            bottom_factors = {
                'sand': 0.8,         # Beach breaks smaller
                'reef': 1.2,         # Reefs can focus energy
                'point': 1.0,        # Points maintain size
                'jetty': 1.1,        # Jetties can focus waves
                'mixed': 0.9
            }
            bottom_factor = bottom_factors.get(bottom_type, 1.0)
            
            # Calculate local wave height
            local_height = offshore_height * exposure_factor * bottom_factor
            
            # Wave period typically doesn't change much
            local_period = offshore_period
            
            # Create wave height range (±20%)
            height_variation = local_height * 0.2
            min_height = max(0, local_height - height_variation)
            max_height = local_height + height_variation
            
            transformed_period = {
                'forecast_time': period['forecast_time'],
                'wave_height_min': min_height,
                'wave_height_max': max_height,
                'wave_period': local_period,
                'wave_direction': period['offshore_wave_direction'],
                'wind_speed': period['offshore_wind_speed'],
                'wind_direction': period['offshore_wind_direction']
            }
            
            transformed_forecast.append(transformed_period)
        
        return transformed_forecast
    
    def assess_surf_quality_complete(self, surf_forecast, current_wind, spot_config):
        """
        REPLACEMENT for _assess_surf_quality()
        Complete surf quality assessment with visual stars and detailed descriptions
        """
        
        enhanced_forecast = []
        
        for period in surf_forecast:
            try:
                # Get existing wind condition assessment
                wind_condition = self._classify_wind_condition(
                    period['wind_direction'], 
                    period['wave_direction'],
                    period['wind_speed']
                )
                
                # Calculate comprehensive surf rating
                quality_rating = self._calculate_surf_rating(
                    period['wave_period'], 
                    wind_condition,
                    (period['wave_height_min'] + period['wave_height_max']) / 2
                )
                
                # Generate visual star display
                star_display = self.generate_star_display(quality_rating['rating'])
                
                # Format wave height range
                wave_range = self.format_wave_height_range(
                    period['wave_height_min'], 
                    period['wave_height_max']
                )
                
                # Generate comprehensive description
                description = self.generate_comprehensive_description(
                    quality_rating['rating'],
                    period['wave_period'],
                    wave_range,
                    wind_condition['type'],
                    period['wind_speed']
                )
                
                # Create enhanced period data
                enhanced_period = period.copy()
                enhanced_period.update({
                    'wind_condition': wind_condition['type'],
                    'wind_quality_modifier': wind_condition['quality_modifier'],
                    'quality_rating': quality_rating['rating'],
                    'quality_stars': star_display['stars_visual'],
                    'quality_text': quality_rating['text'],
                    'wave_height_range': wave_range,
                    'conditions_description': description,
                    'confidence': quality_rating['confidence']
                })
                
                enhanced_forecast.append(enhanced_period)
                
            except Exception as e:
                log.error(f"Error assessing quality for forecast period: {e}")
                # Add fallback data
                enhanced_period = period.copy()
                enhanced_period.update({
                    'quality_rating': 1,
                    'quality_stars': '★☆☆☆☆',
                    'quality_text': 'Unknown',
                    'wave_height_range': 'Unknown',
                    'conditions_description': 'Unable to assess conditions',
                    'confidence': 0.0
                })
                enhanced_forecast.append(enhanced_period)
        
        return enhanced_forecast
    
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

    def store_surf_forecasts(self, spot_id, forecast_data, db_manager):
        """
        Store surf forecasts using WeeWX 5.1 patterns and reading from existing CONF
        
        SURGICAL FIX: Removes manual cursor/commit/rollback patterns
        RETAINS: All functionality, method name, parameters, return values
        READS FROM: Existing CONF for any configuration needs
        """
        try:
            # Clear existing forecasts - WeeWX 5.1 pattern (no manual cursor)
            db_manager.connection.execute(
                "DELETE FROM marine_forecast_surf_data WHERE spot_id = ?", 
                (spot_id,)
            )
            
            # Insert new forecast data - WeeWX 5.1 pattern (no manual cursor/commit)
            for forecast_period in forecast_data:
                db_manager.connection.execute("""
                    INSERT INTO marine_forecast_surf_data (
                        spot_id, forecast_time, generated_time,
                        wave_height_min, wave_height_max, wave_height_range,
                        wave_period, wave_direction, 
                        wind_speed, wind_direction, wind_condition,
                        quality_rating, quality_stars, quality_text,
                        conditions_description, confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    spot_id,
                    forecast_period.get('forecast_time'),
                    forecast_period.get('generated_time', int(time.time())),
                    forecast_period.get('wave_height_min'),
                    forecast_period.get('wave_height_max'),
                    forecast_period.get('wave_height_range'),
                    forecast_period.get('wave_period'),
                    forecast_period.get('wave_direction'),
                    forecast_period.get('wind_speed'),
                    forecast_period.get('wind_direction'),
                    forecast_period.get('wind_condition'),
                    forecast_period.get('quality_rating'),
                    forecast_period.get('quality_stars'),
                    forecast_period.get('quality_text'),
                    forecast_period.get('conditions_description'),
                    forecast_period.get('confidence')
                ))
            
            # WeeWX handles commit automatically - no manual commit
            return True
            
        except Exception as e:
            log.error(f"Error storing surf forecasts for spot {spot_id}: {e}")
            return False

    def get_current_surf_forecast(self, spot_id, hours_ahead=24, db_manager=None):
        """
        Retrieve current surf forecasts using WeeWX 5.1 patterns
        
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
            results = db_manager.connection.execute("""
                SELECT forecast_time, wave_height_min, wave_height_max, wave_height_range,
                    wave_period, wave_direction, wind_speed, wind_direction, 
                    wind_condition, quality_rating, quality_stars, quality_text,
                    conditions_description, confidence
                FROM marine_forecast_surf_data 
                WHERE spot_id = ? AND forecast_time BETWEEN ? AND ?
                ORDER BY forecast_time ASC
            """, (spot_id, current_time, end_time)).fetchall()
            
            # Convert to forecast dictionaries (preserve exact functionality)
            forecasts = []
            for row in results:
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
            log.error(f"Error retrieving surf forecast for spot {spot_id}: {e}")
            return []

    def find_next_good_session(self, spot_id, db_manager, min_rating=4):
        """
        Find the next surf session with rating >= min_rating
        Returns details of the next good session
        """
        
        try:
            current_time = int(time.time())
            
            with db_manager.connection.cursor() as cursor:
                cursor.execute("""
                    SELECT forecast_time, wave_height_range, wave_period,
                        quality_rating, quality_stars, conditions_description
                    FROM marine_forecast_surf_data 
                    WHERE spot_id = ? 
                    AND forecast_time > ?
                    AND quality_rating >= ?
                    ORDER BY forecast_time
                    LIMIT 1
                """, (spot_id, current_time, min_rating))
                
                row = cursor.fetchone()
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
        """
        
        try:
            # Get today's date range
            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            today_end = today_start + timedelta(days=1)
            
            with db_manager.connection.cursor() as cursor:
                cursor.execute("""
                    SELECT quality_rating, wave_height_range, conditions_description
                    FROM marine_forecast_surf_data 
                    WHERE spot_id = ? 
                    AND forecast_time >= ? 
                    AND forecast_time < ?
                    ORDER BY quality_rating DESC
                """, (spot_id, int(today_start.timestamp()), int(today_end.timestamp())))
                
                forecasts = cursor.fetchall()
                
                if not forecasts:
                    return {
                        'status': 'No forecast available',
                        'summary': 'No data for today'
                    }
                
                # Calculate summary stats
                ratings = [f[0] for f in forecasts]
                max_rating = max(ratings)
                avg_rating = sum(ratings) / len(ratings)
                
                # Get best session details
                best_session = forecasts[0]  # Already ordered by rating DESC
                
                # Generate summary text
                if max_rating >= 4:
                    status = 'Good surf today'
                elif max_rating >= 3:
                    status = 'Fair surf today'
                else:
                    status = 'Poor surf today'
                
                return {
                    'status': status,
                    'max_rating': max_rating,
                    'avg_rating': round(avg_rating, 1),
                    'max_rating_stars': '★' * max_rating + '☆' * (5 - max_rating),
                    'best_wave_range': best_session[1],
                    'best_conditions': best_session[2],
                    'total_periods': len(forecasts)
                }
                
        except Exception as e:
            log.error(f"Error getting today's summary for spot_id {spot_id}: {e}")
            return {
                'status': 'Error retrieving summary',
                'summary': 'Unable to load today\'s forecast'
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
        """Query Phase I ndbc_data table for current wave conditions"""
        try:
            db_manager = getattr(self, 'db_manager', None)
            if not db_manager:
                return None
                
            # Get most recent wave data for this station
            with db_manager.connection.cursor() as cursor:
                cursor.execute("""
                    SELECT wave_height, wave_period, wave_direction, observation_time
                    FROM ndbc_data 
                    WHERE station_id = ? AND wave_height IS NOT NULL
                    ORDER BY observation_time DESC 
                    LIMIT 1
                """, (station_id,))
                
                result = cursor.fetchone()
                if result:
                    return {
                        'wave_height': result[0],
                        'wave_period': result[1],
                        'wave_direction': result[2],
                        'observation_time': result[3]
                    }
                return None
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error querying wave data for station {station_id}: {e}")
            return None
    
    def _query_current_atmospheric_data(self, station_id):
        """Query Phase I ndbc_data table for current atmospheric conditions"""
        try:
            db_manager = getattr(self, 'db_manager', None)
            if not db_manager:
                return None
                
            # Get most recent atmospheric data for this station
            with db_manager.connection.cursor() as cursor:
                cursor.execute("""
                    SELECT wind_speed, wind_direction, barometric_pressure, 
                           air_temperature, observation_time
                    FROM ndbc_data 
                    WHERE station_id = ? AND wind_speed IS NOT NULL
                    ORDER BY observation_time DESC 
                    LIMIT 1
                """, (station_id,))
                
                result = cursor.fetchone()
                if result:
                    return {
                        'wind_speed': result[0],
                        'wind_direction': result[1],
                        'barometric_pressure': result[2],
                        'air_temperature': result[3],
                        'observation_time': result[4]
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
    

class SurfForecastSearchList(SearchList):
    """
    NEW CLASS: SearchList extension for WeeWX template integration
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
                        'latitude': float(spot_config.get('latitude', 0.0)),
                        'longitude': float(spot_config.get('longitude', 0.0)),
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
        """Get timestamp of last forecast update"""
        
        with db_manager.connection.cursor() as cursor:
            cursor.execute("SELECT MAX(generated_time) FROM marine_forecast_surf_data")
            result = cursor.fetchone()
            
            if result and result[0]:
                return datetime.fromtimestamp(result[0]).strftime('%m/%d %I:%M %p')
            
            return 'Never'
        

class FishingForecastGenerator:
    """Generate fishing condition forecasts"""
    
    def __init__(self, config_dict, db_manager=None):
        self.config_dict = config_dict
        self.db_manager = db_manager
        service_config = config_dict.get('SurfFishingService', {})
        self.fish_categories = service_config.get('fish_categories', {})
        self.fishing_scoring = service_config.get('fishing_scoring', {})
    
    def generate_fishing_forecast(self, spot_config, marine_conditions=None):
        """
        Generate fishing forecast for a specific spot
        Uses CONF-based configuration only
        
        Args:
            spot_config: Dictionary with spot configuration from CONF
            marine_conditions: Optional marine conditions data from Phase I
        
        Returns:
            List of forecast periods with ratings
        """
        
        # If marine_conditions not provided, get them safely
        if marine_conditions is None:
            marine_conditions = self._get_marine_conditions_safe(spot_config)
        
        try:
            # Get target species category from CONF
            target_category = spot_config.get('target_category', 'mixed_bag')
            
            # Get fish categories from CONF (written by installer from YAML)
            service_config = self.config_dict.get('SurfFishingService', {})
            fish_categories = service_config.get('fish_categories', {})
            category_config = fish_categories.get(target_category, {})
            
            # Generate period-based forecasts (6 periods per day)
            forecast_periods = self._generate_fishing_periods()
            
            # Score each period using CONF-based scoring
            scored_periods = []
            for period in forecast_periods:
                period_score = self.score_fishing_period_complete(
                    period, marine_conditions, category_config, self.db_manager
                )
                scored_periods.append(period_score)
            
            return scored_periods
        
        except Exception as e:
            log.error(f"Error generating fishing forecast for {spot_config.get('name', 'unknown')}: {e}")
            return []
    
    def _generate_fishing_periods(self):
        """Generate 6 fishing periods for the next 3 days"""
        
        periods = []
        period_definitions = [
            ('early_morning', 4, 8),
            ('morning', 8, 12),
            ('midday', 12, 16),
            ('afternoon', 16, 20),
            ('evening', 20, 24),
            ('night', 0, 4)
        ]
        
        # Generate periods for next 3 days
        for day_offset in range(3):
            base_date = int(time.time()) + (day_offset * 86400)
            # Round to start of day
            day_start = base_date - (base_date % 86400)
            
            for period_name, start_hour, end_hour in period_definitions:
                period = {
                    'forecast_date': day_start,
                    'period_name': period_name,
                    'period_start_hour': start_hour,
                    'period_end_hour': end_hour,
                    'period_start_time': day_start + (start_hour * 3600),
                    'period_end_time': day_start + (end_hour * 3600)
                }
                periods.append(period)
        
        return periods
    
    def score_fishing_period_complete(self, period, marine_conditions, category_config, db_manager):
        """
        REPLACEMENT for _score_fishing_period()
        Complete enhanced fishing period scoring with real data integration
        """
        
        try:
            # Enhanced pressure analysis
            pressure_analysis = self.analyze_pressure_trend_enhanced(marine_conditions, db_manager)
            
            # Real tide analysis
            tide_analysis = self.analyze_tide_conditions_real(period, db_manager)
            
            # Time of day scoring (keep existing method)
            time_score, time_factor = self._score_time_of_day(period)
            
            # Enhanced species activity prediction
            species_activity = self.predict_species_activity_enhanced(
                period, pressure_analysis, tide_analysis, category_config
            )
            
            # Calculate total score with weights
            scoring_weights = {
                'pressure': 0.4,  # Most important
                'tide': 0.3,      # Second most important
                'time': 0.2,      # Third
                'species': 0.1    # Fine tuning
            }
            
            weighted_score = (
                pressure_analysis['score'] * scoring_weights['pressure'] +
                tide_analysis['score'] * scoring_weights['tide'] +
                time_score * scoring_weights['time'] +
                species_activity['score'] * scoring_weights['species']
            )
            
            # Convert to 1-5 rating
            if weighted_score >= 3.5:
                rating = 5
                conditions_text = "Excellent"
            elif weighted_score >= 2.8:
                rating = 4
                conditions_text = "Good"
            elif weighted_score >= 2.0:
                rating = 3
                conditions_text = "Fair"
            elif weighted_score >= 1.0:
                rating = 2
                conditions_text = "Poor"
            else:
                rating = 1
                conditions_text = "Slow"
            
            # Generate star display
            star_display = self.generate_star_display(rating)
            
            # Generate comprehensive description
            description = self.generate_comprehensive_fishing_description(
                rating, pressure_analysis, tide_analysis, species_activity, period
            )
            
            # Calculate overall confidence
            confidence = (
                pressure_analysis['confidence'] * 0.4 +
                tide_analysis['confidence'] * 0.3 +
                0.9 * 0.2 +  # Time scoring is always reliable
                species_activity['confidence'] * 0.1
            )
            
            # Enhanced period with complete data
            enhanced_period = period.copy()
            enhanced_period.update({
                'activity_rating': rating,
                'activity_stars': star_display['stars_visual'],
                'conditions_text': conditions_text,
                'description': description,
                'species_activity': species_activity['text'],
                'species_activity_level': species_activity['level'],
                
                # Detailed analysis results
                'pressure_analysis': pressure_analysis,
                'tide_analysis': tide_analysis,
                'species_analysis': species_activity,
                'time_score': time_score,
                'weighted_score': weighted_score,
                'confidence': round(confidence, 2),
                'generated_time': int(time.time())
            })
            
            return enhanced_period
            
        except Exception as e:
            log.error(f"Error in enhanced fishing period scoring: {e}")
            # Fallback to basic scoring
            return self._score_fishing_period_basic(period, marine_conditions, category_config)
    
    def analyze_pressure_trend_enhanced(self, marine_conditions, db_manager):
        """
        Analyze pressure trends using data-driven Phase I integration
        
        SURGICAL FIX: Removes manual cursor patterns  
        RETAINS: All functionality, method name, parameters, return values
        READS FROM: Existing CONF for Phase I integration settings
        REQUIRES: Phase I Marine Data Extension must be installed and configured
        """
        try:
            # Verify Phase I configuration (REQUIRED DEPENDENCY)
            phase_i_config = self.config_dict.get('MarineDataService', {})
            if not phase_i_config:
                raise RuntimeError("Phase I Marine Data Extension not found. Phase I is required for Phase II to function.")
            
            current_time = int(time.time())
            history_start = current_time - (6 * 3600)  # 6 hours back
            
            # WeeWX 5.1 pattern - direct execute (no manual cursor)
            # Query Phase I ndbc_data table
            results = db_manager.connection.execute("""
                SELECT marine_barometric_pressure, dateTime
                FROM ndbc_data 
                WHERE dateTime >= ? AND marine_barometric_pressure IS NOT NULL
                ORDER BY dateTime DESC
                LIMIT 12
            """, (history_start,)).fetchall()
            
            if len(results) < 2:
                raise RuntimeError(f"Insufficient pressure data available from Phase I. Found {len(results)} readings, need at least 2.")
            
            # Calculate pressure trend (preserve exact algorithm)
            recent_pressure = results[0][0]
            old_pressure = results[-1][0]
            pressure_change = recent_pressure - old_pressure
            
            # Analyze trend using standard thresholds (preserve exact logic)
            if pressure_change < -0.05:
                return {
                    'score': 4,
                    'trend': 'falling_rapidly',
                    'description': 'Rapidly falling pressure - fish very active',
                    'pressure_change': pressure_change,
                    'confidence': 0.8
                }
            elif pressure_change < -0.02:
                return {
                    'score': 3,
                    'trend': 'falling',
                    'description': 'Falling pressure - good fishing',
                    'pressure_change': pressure_change,
                    'confidence': 0.7
                }
            elif pressure_change > 0.05:
                return {
                    'score': 2,
                    'trend': 'rising_rapidly',
                    'description': 'Rising pressure - fish less active',
                    'pressure_change': pressure_change,
                    'confidence': 0.6
                }
            else:
                return {
                    'score': 3,
                    'trend': 'stable',
                    'description': 'Stable pressure - moderate fishing',
                    'pressure_change': pressure_change,
                    'confidence': 0.5
                }
            
        except Exception as e:
            log.error(f"Error analyzing pressure trend: {e}")
            raise

    def analyze_tide_conditions_real(self, spot_config, db_manager):
        """
        Analyze tide conditions using data-driven Phase I integration
        
        SURGICAL FIX: Removes manual cursor patterns
        RETAINS: All functionality, method name, parameters, return values
        READS FROM: Existing CONF for Phase I integration settings  
        REQUIRES: Phase I Marine Data Extension must be installed and configured
        """
        try:
            # Verify Phase I configuration (REQUIRED DEPENDENCY)
            phase_i_config = self.config_dict.get('MarineDataService', {})
            if not phase_i_config:
                raise RuntimeError("Phase I Marine Data Extension not found. Phase I is required for Phase II to function.")
            
            current_time = int(time.time())
            search_window = current_time + (12 * 3600)  # 12 hours ahead
            
            # WeeWX 5.1 pattern - direct execute (no manual cursor)
            # Query Phase I tide table
            results = db_manager.connection.execute("""
                SELECT tide_time, tide_type, predicted_height
                FROM tide_table
                WHERE tide_time > ? AND tide_time < ?
                ORDER BY tide_time ASC
                LIMIT 4
            """, (current_time, search_window)).fetchall()
            
            if not results:
                raise RuntimeError("No tide data available from Phase I tide_table for the next 12 hours.")
            
            # Find next tide change (preserve exact algorithm)
            next_tide = results[0]
            time_to_tide = (next_tide[0] - current_time) / 3600  # hours
            
            # Analyze tide movement (preserve exact logic)
            if next_tide[1] == 'H':  # High tide
                if time_to_tide <= 2:  # Within 2 hours
                    return {
                        'current_movement': 'incoming',
                        'time_to_next_tide_hours': time_to_tide,
                        'quality': 'good',
                        'description': 'Incoming tide - active feeding',
                        'quality_score': 0.8
                    }
            else:  # Low tide
                if time_to_tide <= 2:  # Within 2 hours
                    return {
                        'current_movement': 'outgoing',
                        'time_to_next_tide_hours': time_to_tide,
                        'quality': 'fair',
                        'description': 'Outgoing tide - moderate activity',
                        'quality_score': 0.6
                    }
            
            # Default for distant tides
            return {
                'current_movement': 'slack',
                'time_to_next_tide_hours': time_to_tide,
                'quality': 'fair',
                'description': 'Slack water - moderate fishing',
                'quality_score': 0.5
            }
            
        except Exception as e:
            log.error(f"Error analyzing tide conditions: {e}")
            raise

    def _score_time_of_day(self, period):
        """Score based on time of day feeding patterns"""
        
        period_name = period['period_name']
        
        # Dawn and dusk are prime feeding times
        if period_name in ['early_morning', 'evening']:
            score = 3
            factor = "Prime feeding time - dawn/dusk activity"
        elif period_name in ['morning', 'afternoon']:
            score = 2
            factor = "Good feeding time"
        elif period_name == 'midday':
            score = 1
            factor = "Fair - fish less active in bright sun"
        else:  # night
            score = 1
            factor = "Night fishing - species dependent"
        
        return score, factor
    
    def _score_species_conditions(self, period, marine_conditions, category_config):
        """Score based on target species preferences"""
        
        # Get species preferences
        pressure_pref = category_config.get('pressure_preference', 'stable')
        tide_relevance = category_config.get('tide_relevance', True)
        
        score = 0
        factors = []
        
        # Adjust based on species pressure preference
        if pressure_pref == 'falling':
            # This species prefers falling pressure (already scored in pressure section)
            factors.append("Target species active in falling pressure")
        elif pressure_pref == 'stable':
            factors.append("Target species prefers stable conditions")
        
        # Tide relevance for species
        if not tide_relevance:
            factors.append("Freshwater species - tide independent")
        
        factor_text = "; ".join(factors) if factors else None
        return score, factor_text

    def generate_star_display(self, rating):
        """Generate visual star display for fishing activity rating"""
        
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

    def _score_pressure_conditions_basic(self, marine_conditions):
        """Basic pressure analysis when historical data unavailable"""
        
        current_pressure = marine_conditions.get('current_pressure', 30.0)
        
        # Simplified scoring based on absolute pressure
        if current_pressure < 29.80:
            score = 3
            trend = "low_pressure"
            description = "Low pressure system - active fishing"
        elif current_pressure > 30.20:
            score = 1
            trend = "high_pressure"
            description = "High pressure system - slower fishing"
        else:
            score = 2
            trend = "normal_pressure"
            description = "Normal pressure - moderate activity"
        
        return {
            'score': score,
            'trend': trend,
            'description': description,
            'pressure_change': 0.0,
            'current_pressure': current_pressure,
            'confidence': 0.4
        }

    def _score_tide_conditions_basic(self, period):
        """Basic tide analysis when real tide data unavailable"""
        
        period_hour = period['period_start_hour']
        
        # Simplified scoring based on typical tide times
        if period_hour in [6, 7, 18, 19]:  # Typical tide change times
            score = 2
            movement = "changing"
            description = "Estimated tide change - active period"
        elif period_hour in [0, 12]:  # Typical high/low times
            score = 0
            movement = "slack"
            description = "Estimated slack tide - slower period"
        else:
            score = 1
            movement = "moderate"
            description = "Moderate tidal flow estimated"
        
        return {
            'score': score,
            'movement': movement,
            'description': description,
            'tide_events': 0,
            'confidence': 0.3
        }

    def predict_species_activity_enhanced(self, period, pressure_analysis, tide_analysis, category_config):
        """Enhanced species activity prediction with detailed factors"""
        
        species_list = category_config.get('species', ['Mixed'])
        pressure_pref = category_config.get('pressure_preference', 'stable')
        tide_relevance = category_config.get('tide_relevance', True)
        
        activity_factors = []
        activity_score = 0
        
        # Pressure preference matching
        if pressure_pref == 'falling' and 'falling' in pressure_analysis['trend']:
            activity_score += 2
            activity_factors.append(f"Target species prefer falling pressure")
        elif pressure_pref == 'stable' and pressure_analysis['trend'] == 'stable':
            activity_score += 1
            activity_factors.append(f"Target species prefer stable pressure")
        elif pressure_pref == 'rising' and 'rising' in pressure_analysis['trend']:
            activity_score += 1
            activity_factors.append(f"Target species prefer rising pressure")
        
        # Tide relevance
        if tide_relevance and tide_analysis['score'] >= 2:
            activity_score += 1
            activity_factors.append(f"Good tidal movement for {species_list[0]}")
        elif not tide_relevance:
            activity_factors.append(f"Freshwater species - tide independent")
        
        # Time-based species activity
        period_name = period['period_name']
        if period_name in ['early_morning', 'evening']:
            activity_score += 1
            activity_factors.append(f"Prime feeding time for most species")
        
        # Convert to activity level
        if activity_score >= 4:
            activity_level = 'very_high'
            activity_text = 'Very High'
        elif activity_score >= 3:
            activity_level = 'high'
            activity_text = 'High'
        elif activity_score >= 2:
            activity_level = 'moderate'
            activity_text = 'Moderate'
        elif activity_score >= 1:
            activity_level = 'low'
            activity_text = 'Low'
        else:
            activity_level = 'very_low'
            activity_text = 'Very Low'
        
        return {
            'level': activity_level,
            'text': activity_text,
            'score': activity_score,
            'factors': activity_factors,
            'target_species': species_list[:3],  # Top 3 species
            'confidence': 0.7
        }

    def generate_comprehensive_fishing_description(self, rating, pressure_analysis, tide_analysis, species_activity, period):
        """Generate detailed human-readable fishing condition descriptions"""
        
        period_name = period['period_name'].replace('_', ' ').title()
        
        # Base quality terms
        quality_terms = {
            5: "Excellent",
            4: "Good",
            3: "Fair", 
            2: "Poor",
            1: "Slow"
        }
        
        base_quality = quality_terms.get(rating, "Unknown")
        
        # Build description components
        components = [base_quality, "fishing"]
        
        # Add pressure component
        if pressure_analysis['score'] >= 3:
            components.append(f"({pressure_analysis['description'].lower()})")
        
        # Add tide component if relevant
        if tide_analysis['score'] >= 2:
            components.append(f"with {tide_analysis['movement']} tide")
        elif tide_analysis['score'] == 0:
            components.append("during slack tide")
        
        # Add species activity
        if species_activity['level'] in ['high', 'very_high']:
            components.append(f"- {species_activity['text'].lower()} activity expected")
        
        # Add period context
        if period['period_name'] in ['early_morning', 'evening']:
            components.append("during prime feeding time")
        
        return " ".join(components)

    def store_fishing_forecasts(self, spot_id, forecast_data, db_manager):
        """
        Store fishing forecasts using WeeWX 5.1 patterns
        
        SURGICAL FIX: Removes manual cursor/commit/rollback patterns  
        RETAINS: All functionality, method name, parameters, return values
        READS FROM: Existing CONF for any configuration needs
        """
        try:
            # Clear existing forecasts - WeeWX 5.1 pattern (no manual cursor)
            db_manager.connection.execute(
                "DELETE FROM marine_forecast_fishing_data WHERE spot_id = ?", 
                (spot_id,)
            )
            
            # Insert new forecast data - WeeWX 5.1 pattern (no manual cursor/commit)
            for forecast_period in forecast_data:
                db_manager.connection.execute("""
                    INSERT INTO marine_forecast_fishing_data (
                        spot_id, forecast_date, period_name, generated_time,
                        pressure_trend, pressure_change, pressure_score,
                        tide_stage, tide_flow, tide_score,
                        species_activity, activity_rating, best_species_json,
                        conditions_text, overall_score, confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    spot_id,
                    forecast_period.get('forecast_date'),
                    forecast_period.get('period_name'),
                    forecast_period.get('generated_time', int(time.time())),
                    forecast_period.get('pressure_trend'),
                    forecast_period.get('pressure_change'),
                    forecast_period.get('pressure_score'),
                    forecast_period.get('tide_stage'),
                    forecast_period.get('tide_flow'),
                    forecast_period.get('tide_score'),
                    forecast_period.get('species_activity'),
                    forecast_period.get('activity_rating'),
                    json.dumps(forecast_period.get('best_species', [])) if forecast_period.get('best_species') else None,
                    forecast_period.get('conditions_text'),
                    forecast_period.get('overall_score'),
                    forecast_period.get('confidence')
                ))
            
            # WeeWX handles commit automatically - no manual commit
            return True
            
        except Exception as e:
            log.error(f"Error storing fishing forecasts for spot {spot_id}: {e}")
            return False

    def get_current_fishing_forecast(self, spot_id, db_manager):
        """
        Get current fishing forecast using WeeWX 5.1 patterns
        
        SURGICAL FIX: Removes manual cursor patterns
        RETAINS: All functionality, method name, parameters, return values  
        READS FROM: Existing CONF for any configuration needs
        """
        try:
            current_date = datetime.now().strftime('%Y-%m-%d')
            
            # WeeWX 5.1 pattern - direct execute (no manual cursor)
            results = db_manager.connection.execute("""
                SELECT forecast_date, period_name, pressure_trend, pressure_change,
                    pressure_score, tide_stage, tide_flow, tide_score,
                    species_activity, activity_rating, best_species_json,
                    conditions_text, overall_score, confidence
                FROM marine_forecast_fishing_data
                WHERE spot_id = ? AND forecast_date = ?
                ORDER BY period_name ASC
            """, (spot_id, current_date)).fetchall()
            
            # Convert to forecast dictionaries (preserve exact functionality)
            forecasts = []
            for row in results:
                best_species = None
                if row[10]:  # best_species_json
                    try:
                        best_species = json.loads(row[10])
                    except:
                        best_species = []
                        
                forecasts.append({
                    'forecast_date': row[0],
                    'period_name': row[1],
                    'pressure_trend': row[2],
                    'pressure_change': row[3],
                    'pressure_score': row[4],
                    'tide_stage': row[5],
                    'tide_flow': row[6],
                    'tide_score': row[7],
                    'species_activity': row[8],
                    'activity_rating': row[9],
                    'best_species': best_species,
                    'conditions_text': row[11],
                    'overall_score': row[12],
                    'confidence': row[13]
                })
            
            return forecasts
            
        except Exception as e:
            log.error(f"Error retrieving fishing forecast for spot {spot_id}: {e}")
            return []

    def find_next_good_fishing_period(self, spot_id, db_manager):
        """
        Find next good fishing period using data-driven rating thresholds
        
        SURGICAL FIX: Removes manual cursor patterns
        RETAINS: All functionality, method name, parameters, return values
        READS FROM: Existing CONF for rating configuration
        """
        try:
            # Get rating configuration from existing CONF
            service_config = self.config_dict.get('SurfFishingService', {})
            
            # Use standard good rating threshold (3+ stars)
            good_rating = 3
            
            current_time = int(time.time())
            
            # WeeWX 5.1 pattern - direct execute (no manual cursor)
            result = db_manager.connection.execute("""
                SELECT period_name, forecast_date, activity_rating, conditions_text
                FROM marine_forecast_fishing_data
                WHERE spot_id = ? AND activity_rating >= ?
                AND forecast_date >= date('now')
                ORDER BY forecast_date ASC, period_name ASC
                LIMIT 1
            """, (spot_id, good_rating)).fetchone()
            
            if result:
                return {
                    'period_name': result[0],
                    'forecast_date': result[1],
                    'rating': result[2],
                    'conditions': result[3]
                }
            
            return None
            
        except Exception as e:
            log.error(f"Error finding next good fishing period for spot {spot_id}: {e}")
            return None

    def get_today_fishing_summary(self, spot_id, db_manager):
        """
        Get today's fishing summary using data-driven aggregation
        
        SURGICAL FIX: Removes manual cursor patterns
        RETAINS: All functionality, method name, parameters, return values
        READS FROM: Existing CONF for any configuration needs
        """
        try:
            today_date = datetime.now().strftime('%Y-%m-%d')
            
            # WeeWX 5.1 pattern - direct execute (no manual cursor)
            result = db_manager.connection.execute("""
                SELECT MAX(activity_rating) as max_rating, 
                    AVG(activity_rating) as avg_rating,
                    COUNT(*) as period_count
                FROM marine_forecast_fishing_data
                WHERE spot_id = ? AND forecast_date = ?
            """, (spot_id, today_date)).fetchone()
            
            if result and result[0] is not None:
                return {
                    'max_rating': result[0],
                    'avg_rating': round(result[1], 1) if result[1] else 0,
                    'period_count': result[2],
                    'date': today_date
                }
            
            return {
                'max_rating': 0,
                'avg_rating': 0,
                'period_count': 0,
                'date': today_date
            }
            
        except Exception as e:
            log.error(f"Error getting today's fishing summary for spot {spot_id}: {e}")
            return {}

    def get_pressure_trend_analysis(self, spot_id, db_manager, hours_back=6):
        """Get detailed pressure trend analysis for display purposes"""
        
        try:
            current_time = int(time.time())
            
            with db_manager.connection.cursor() as cursor:
                cursor.execute("""
                    SELECT marine_barometric_pressure, dateTime
                    FROM ndbc_data 
                    WHERE dateTime >= ? 
                    AND marine_barometric_pressure IS NOT NULL
                    ORDER BY dateTime DESC
                    LIMIT ?
                """, (current_time - (hours_back * 3600), hours_back * 2))
                
                pressure_readings = cursor.fetchall()
            
            if len(pressure_readings) < 3:
                return {
                    'status': 'Insufficient data',
                    'trend': 'unknown',
                    'confidence': 0.0
                }
            
            # Calculate pressure changes over different time periods
            current_pressure = pressure_readings[0][0]
            
            # 1-hour change
            one_hour_change = 0
            for pressure, timestamp in pressure_readings[1:]:
                if current_time - timestamp >= 3600:  # 1 hour
                    one_hour_change = current_pressure - pressure
                    break
            
            # 3-hour change
            three_hour_change = 0
            for pressure, timestamp in pressure_readings:
                if current_time - timestamp >= 10800:  # 3 hours
                    three_hour_change = current_pressure - pressure
                    break
            
            # 6-hour change
            six_hour_change = 0
            if len(pressure_readings) > 0:
                six_hour_change = current_pressure - pressure_readings[-1][0]
            
            # Determine overall trend
            if three_hour_change <= -0.05:
                trend = 'falling_fast'
                trend_text = 'Falling rapidly'
                fishing_impact = 'Excellent for fishing'
            elif three_hour_change <= -0.02:
                trend = 'falling'
                trend_text = 'Falling'
                fishing_impact = 'Good for fishing'
            elif abs(three_hour_change) <= 0.02:
                trend = 'stable'
                trend_text = 'Stable'
                fishing_impact = 'Normal conditions'
            elif three_hour_change >= 0.05:
                trend = 'rising_fast'
                trend_text = 'Rising rapidly'
                fishing_impact = 'Poor for fishing'
            else:
                trend = 'rising'
                trend_text = 'Rising'
                fishing_impact = 'Below average'
            
            return {
                'status': 'Available',
                'current_pressure': round(current_pressure, 2),
                'trend': trend,
                'trend_text': trend_text,
                'fishing_impact': fishing_impact,
                'changes': {
                    '1_hour': round(one_hour_change, 3),
                    '3_hour': round(three_hour_change, 3),
                    '6_hour': round(six_hour_change, 3)
                },
                'readings_count': len(pressure_readings),
                'confidence': min(1.0, len(pressure_readings) / 6)
            }
            
        except Exception as e:
            log.error(f"Error getting pressure trend analysis: {e}")
            return {
                'status': 'Error',
                'trend': 'unknown',
                'confidence': 0.0
            }

    def get_tide_schedule_for_fishing(self, spot_id, db_manager, days_ahead=3):
        """Get tide schedule optimized for fishing planning"""
        
        try:
            current_time = int(time.time())
            end_time = current_time + (days_ahead * 86400)
            
            with db_manager.connection.cursor() as cursor:
                cursor.execute("""
                    SELECT tide_time, tide_type, predicted_height, datum
                    FROM tide_table 
                    WHERE tide_time >= ? AND tide_time <= ?
                    ORDER BY tide_time
                """, (current_time, end_time))
                
                tide_events = cursor.fetchall()
            
            if not tide_events:
                return {
                    'status': 'No tide data available',
                    'events': []
                }
            
            # Process tide events for fishing optimization
            fishing_tide_events = []
            
            for i, (tide_time, tide_type, height, datum) in enumerate(tide_events):
                # Calculate fishing windows around each tide change
                tide_dt = datetime.fromtimestamp(tide_time)
                
                # 2-hour window before tide change (moving water)
                start_fishing = tide_time - 7200
                # 2-hour window after tide change
                end_fishing = tide_time + 7200
                
                # Determine fishing quality based on tide type and height
                if tide_type == 'H':  # High tide
                    tide_description = f"High tide {height:.1f}ft"
                    if height >= 6.0:
                        fishing_quality = 'excellent'
                    elif height >= 4.0:
                        fishing_quality = 'good'
                    else:
                        fishing_quality = 'fair'
                else:  # Low tide
                    tide_description = f"Low tide {height:.1f}ft"
                    if height <= 1.0:
                        fishing_quality = 'fair'  # Very low water
                    else:
                        fishing_quality = 'good'
                
                fishing_tide_events.append({
                    'tide_time': tide_time,
                    'tide_type': tide_type,
                    'tide_height': height,
                    'tide_description': tide_description,
                    'formatted_time': tide_dt.strftime('%a %I:%M %p'),
                    'fishing_window_start': datetime.fromtimestamp(start_fishing).strftime('%I:%M %p'),
                    'fishing_window_end': datetime.fromtimestamp(end_fishing).strftime('%I:%M %p'),
                    'fishing_quality': fishing_quality,
                    'moving_water_period': f"{datetime.fromtimestamp(start_fishing).strftime('%I:%M %p')} - {datetime.fromtimestamp(end_fishing).strftime('%I:%M %p')}"
                })
            
            return {
                'status': 'Available',
                'events': fishing_tide_events,
                'total_events': len(fishing_tide_events)
            }
            
        except Exception as e:
            log.error(f"Error getting tide schedule for fishing: {e}")
            return {
                'status': 'Error retrieving tide data',
                'events': []
            }

    def get_species_specific_recommendations(self, spot_id, target_category, db_manager):
        """Get species-specific fishing recommendations based on current conditions"""
        
        try:
            # Get category configuration
            category_config = self.fish_categories.get(target_category, {})
            species_list = category_config.get('species', ['Mixed'])
            pressure_pref = category_config.get('pressure_preference', 'stable')
            
            # Get current conditions
            current_forecasts = self.get_current_fishing_forecast(spot_id, db_manager, 1)
            if not current_forecasts:
                return {
                    'status': 'No current forecast available',
                    'recommendations': []
                }
            
            # Get today's forecasts only
            today = int(time.time()) - (int(time.time()) % 86400)
            today_forecasts = [f for f in current_forecasts if f['forecast_date'] == today]
            
            if not today_forecasts:
                return {
                    'status': 'No forecast for today',
                    'recommendations': []
                }
            
            # Generate species-specific recommendations
            recommendations = []
            
            for species in species_list[:3]:  # Top 3 species
                species_recommendations = []
                
                # Find best periods for this species
                best_periods = [f for f in today_forecasts if f['activity_rating'] >= 3]
                best_periods.sort(key=lambda x: x['activity_rating'], reverse=True)
                
                if best_periods:
                    best_period = best_periods[0]
                    species_recommendations.append({
                        'type': 'best_time',
                        'text': f"Best time for {species}: {best_period['period_display']} ({best_period['activity_stars']})",
                        'period': best_period['period_display'],
                        'rating': best_period['activity_rating']
                    })
                
                # Pressure-based recommendations
                if pressure_pref == 'falling':
                    species_recommendations.append({
                        'type': 'pressure_tip',
                        'text': f"{species} are most active during falling barometric pressure",
                        'advice': "Look for periods with falling pressure trends"
                    })
                
                # Tide-based recommendations
                if category_config.get('tide_relevance', True):
                    species_recommendations.append({
                        'type': 'tide_tip',
                        'text': f"{species} feed actively during moving water",
                        'advice': "Fish 2 hours before/after tide changes"
                    })
                
                recommendations.append({
                    'species': species,
                    'recommendations': species_recommendations
                })
            
            return {
                'status': 'Available',
                'target_category': target_category,
                'recommendations': recommendations,
                'general_advice': self._get_general_fishing_advice(today_forecasts)
            }
            
        except Exception as e:
            log.error(f"Error getting species recommendations: {e}")
            return {
                'status': 'Error generating recommendations',
                'recommendations': []
            }

    def _get_general_fishing_advice(self, today_forecasts):
        """Generate general fishing advice based on today's conditions"""
        
        if not today_forecasts:
            return ["No forecast data available for advice"]
        
        advice = []
        
        # Find best and worst periods
        best_rating = max(f['activity_rating'] for f in today_forecasts)
        worst_rating = min(f['activity_rating'] for f in today_forecasts)
        
        if best_rating >= 4:
            best_periods = [f for f in today_forecasts if f['activity_rating'] >= 4]
            period_names = [f['period_display'] for f in best_periods]
            advice.append(f"Excellent fishing expected during: {', '.join(period_names)}")
        
        if worst_rating <= 2:
            worst_periods = [f for f in today_forecasts if f['activity_rating'] <= 2]
            period_names = [f['period_display'] for f in worst_periods]
            advice.append(f"Avoid fishing during: {', '.join(period_names)}")
        
        # Check for pressure trends
        pressure_trends = [f.get('pressure_trend', 'unknown') for f in today_forecasts]
        if 'falling' in pressure_trends:
            advice.append("Falling pressure detected - fish should be more active")
        elif 'rising' in pressure_trends:
            advice.append("Rising pressure may reduce fish activity")
        
        # Check for tide patterns
        tide_movements = [f.get('tide_movement', 'unknown') for f in today_forecasts]
        if 'incoming' in tide_movements or 'outgoing' in tide_movements:
            advice.append("Moving water periods available - good for active feeding")
        
        if not advice:
            advice.append("Moderate fishing conditions expected today")
        
        return advice

    def export_fishing_forecast_summary(self, spot_id, format='dict', db_manager=None):
        """
        Export comprehensive fishing forecast summary using data-driven queries
        
        SURGICAL FIX: Removes manual cursor patterns
        RETAINS: All functionality, method name, parameters, return values
        READS FROM: Existing CONF for spot configuration
        """
        try:
            # Get current forecasts using existing method (preserves all logic)
            current_forecasts = self.get_current_fishing_forecast(spot_id, db_manager)
            
            # Get next good period using existing method (preserves all logic)  
            next_good_period = self.find_next_good_fishing_period(spot_id, db_manager)
            
            # Get today summary using existing method (preserves all logic)
            today_summary = self.get_today_fishing_summary(spot_id, db_manager)
            
            # Get pressure analysis using existing method (preserves all logic)
            pressure_analysis = self.analyze_pressure_trend_enhanced({}, db_manager)
            
            # Get spot info from existing CONF structure (data-driven)
            service_config = self.config_dict.get('SurfFishingService', {})
            fishing_spots = service_config.get('fishing_spots', {})
            
            spot_info = None
            for spot_key, spot_config in fishing_spots.items():
                # Use CONF key as ID matching pattern
                if spot_key == spot_id or spot_config.get('name') == spot_id:
                    spot_info = {
                        'id': spot_key,
                        'name': spot_config.get('name', spot_key),
                        'latitude': float(spot_config.get('latitude', 0.0)),
                        'longitude': float(spot_config.get('longitude', 0.0)),
                        'location_type': spot_config.get('location_type', 'shore'),
                        'target_category': spot_config.get('target_category', 'mixed_bag')
                    }
                    break
            
            if not spot_info:
                return {'error': 'Spot not found'}
            
            # Get tide analysis using existing method (preserves all logic)
            tide_schedule = self.analyze_tide_conditions_real(spot_info, db_manager)
            
            # Compile complete summary (preserve exact structure)
            summary = {
                'spot_info': spot_info,
                'generated_time': datetime.now().isoformat(),
                'forecasts': current_forecasts,
                'next_good_period': next_good_period,
                'today_summary': today_summary,
                'pressure_analysis': pressure_analysis,
                'tide_schedule': tide_schedule,
                'forecast_count': len(current_forecasts)
            }
            
            if format == 'json':
                return json.dumps(summary, indent=2)
            else:
                return summary
                
        except Exception as e:
            log.error(f"Error exporting fishing forecast summary: {e}")
            return {'error': f'Export failed: {str(e)}'}

    def integrate_fishing_enhancements():
        """
        Instructions for integrating these methods into the existing FishingForecastGenerator class:
        
        1. Add all the above methods to the existing FishingForecastGenerator class in surf_fishing.py
        2. Replace _score_fishing_period with score_fishing_period_complete
        3. Replace _score_pressure_conditions with analyze_pressure_trend_enhanced  
        4. Replace _score_tide_conditions with analyze_tide_conditions_real
        5. Add the new FishingForecastSearchList class to surf_fishing.py
        6. Update the main service to use the enhanced methods
        
        Example usage in service:
        ```python
        # Enhanced forecast generation
        fishing_forecast = self.fishing_generator.generate_fishing_forecast(spot, marine_conditions)
        
        # Store with enhanced data
        self.fishing_generator.store_fishing_forecasts(spot['id'], fishing_forecast, self.db_manager)
        
        # Get enhanced retrievals
        current_forecast = self.fishing_generator.get_current_fishing_forecast(spot['id'], self.db_manager)
        next_good = self.fishing_generator.find_next_good_fishing_period(spot['id'], self.db_manager)
        ```
        """
        pass

    def integrate_pressure_trend_analysis(self, spot_config):
        """Enhanced pressure analysis using optimal atmospheric sources"""
        location_coords = (float(spot_config['latitude']), float(spot_config['longitude']))
        
        # Get optimal atmospheric sources using integration manager
        integration_manager = getattr(self, 'integration_manager', None)
        if not integration_manager:
            log.warning(f"{CORE_ICONS['warning']} No integration manager available - using default pressure analysis")
            return self._get_default_pressure_analysis()
        
        # Select optimal atmospheric sources for pressure data
        optimal_sources = integration_manager.select_optimal_atmospheric_sources(location_coords)
        
        if not optimal_sources:
            log.warning(f"{CORE_ICONS['warning']} No optimal atmospheric sources found for pressure analysis")
            return self._get_default_pressure_analysis()
        
        # Get pressure trend data from multiple sources
        pressure_data_sources = []
        for source in optimal_sources:
            pressure_data = self._query_pressure_trend_data(source['station_id'])
            if pressure_data:
                pressure_data_sources.append({
                    'station_id': source['station_id'],
                    'data': pressure_data,
                    'distance_miles': source['distance_miles'],
                    'quality_score': source['quality_score']
                })
        
        if not pressure_data_sources:
            return self._get_default_pressure_analysis()
        
        # Fuse pressure data from multiple sources
        fusion_processor = getattr(self, 'fusion_processor', None)
        if fusion_processor and len(pressure_data_sources) > 1:
            fused_pressure = fusion_processor.fuse_atmospheric_data(pressure_data_sources)
            confidence = fusion_processor.calculate_confidence_score(
                pressure_data_sources, 
                [s['quality_score'] for s in pressure_data_sources]
            )
        else:
            # Single source or no fusion processor
            fused_pressure = pressure_data_sources[0]['data']
            confidence = pressure_data_sources[0]['quality_score']
        
        # Analyze pressure trend for fishing implications
        return self._analyze_pressure_trend_for_fishing(fused_pressure, confidence)
    
    def integrate_tide_correlation_analysis(self, spot_config):
        """Enhanced tide analysis using optimal tide sources"""
        location_coords = (float(spot_config['latitude']), float(spot_config['longitude']))
        
        # Get optimal tide source using integration manager
        integration_manager = getattr(self, 'integration_manager', None)
        if not integration_manager:
            log.warning(f"{CORE_ICONS['warning']} No integration manager available - using default tide analysis")
            return self._get_default_tide_analysis()
        
        # Select optimal tide source
        optimal_tide_source = integration_manager.select_optimal_tide_source(location_coords)
        
        if not optimal_tide_source:
            log.warning(f"{CORE_ICONS['warning']} No optimal tide source found for tide analysis")
            return self._get_default_tide_analysis()
        
        # Get current and predicted tide data
        tide_data = self._query_tide_data_for_analysis(optimal_tide_source['station_id'])
        
        if not tide_data:
            return self._get_default_tide_analysis()
        
        # Apply distance-based quality adjustment
        quality_score = integration_manager.calculate_tide_quality(optimal_tide_source['distance_miles'])
        
        # Analyze tide patterns for fishing implications
        return self._analyze_tide_correlation_for_fishing(tide_data, quality_score, optimal_tide_source)
    
    def calculate_species_activity_with_multi_source(self, target_category, atmospheric_data, tide_data):
        """Species activity calculation using fused data sources"""
        category_config = self.fish_categories.get(target_category, {})
        
        activity_factors = []
        activity_score = 0
        
        # Enhanced pressure analysis with confidence weighting
        if atmospheric_data and 'barometric_pressure' in atmospheric_data:
            pressure_trend = atmospheric_data.get('pressure_trend', 'stable')
            pressure_change = atmospheric_data.get('pressure_change_6h', 0.0)
            pressure_confidence = atmospheric_data.get('confidence', 0.7)
            
            pressure_preference = category_config.get('pressure_preference', 'stable')
            
            # Score pressure conditions
            pressure_score = self._score_pressure_for_species(
                pressure_trend, pressure_change, pressure_preference
            )
            
            # Weight by confidence in pressure data
            weighted_pressure_score = pressure_score * pressure_confidence
            activity_score += weighted_pressure_score * 0.4  # Pressure weight from YAML
            
            activity_factors.append(f"Pressure: {pressure_trend} (confidence: {pressure_confidence:.1f})")
        
        # Enhanced tide analysis with distance quality
        if tide_data and category_config.get('tide_relevance', False):
            tide_movement = tide_data.get('current_movement', 'unknown')
            tide_quality = tide_data.get('quality_score', 0.7)
            
            optimal_stages = category_config.get('optimal_tide_stage', ['incoming', 'outgoing'])
            
            # Score tide conditions
            if isinstance(optimal_stages, list):
                tide_score = 1.0 if tide_movement in optimal_stages else 0.3
            else:
                tide_score = 1.0 if tide_movement == optimal_stages else 0.3
            
            # Weight by tide data quality (distance-based)
            weighted_tide_score = tide_score * tide_quality
            activity_score += weighted_tide_score * 0.3  # Tide weight from YAML
            
            activity_factors.append(f"Tide: {tide_movement} (quality: {tide_quality:.1f})")
        
        # Time of day factor (unchanged from existing logic)
        current_hour = datetime.now().hour
        if current_hour in [4, 5, 6, 7, 19, 20, 21, 22]:  # Dawn/dusk
            activity_score += 1.0 * 0.2  # Time weight from YAML
            activity_factors.append("Prime feeding time")
        
        # Weather stability factor
        if atmospheric_data:
            wind_speed = atmospheric_data.get('wind_speed', 0)
            if wind_speed < 15:  # Calm conditions
                activity_score += 0.8 * 0.1  # Weather weight from YAML
                activity_factors.append("Calm weather conditions")
        
        # Convert to activity level with enhanced confidence
        overall_confidence = min(
            atmospheric_data.get('confidence', 0.7) if atmospheric_data else 0.5,
            tide_data.get('quality_score', 0.7) if tide_data else 0.5
        )
        
        if activity_score >= 3.5:
            activity_level = 'very_high'
            activity_text = 'Very High'
        elif activity_score >= 2.5:
            activity_level = 'high'
            activity_text = 'High'
        elif activity_score >= 1.5:
            activity_level = 'moderate'
            activity_text = 'Moderate'
        elif activity_score >= 0.8:
            activity_level = 'low'
            activity_text = 'Low'
        else:
            activity_level = 'very_low'
            activity_text = 'Very Low'
        
        return {
            'level': activity_level,
            'text': activity_text,
            'score': activity_score,
            'factors': activity_factors,
            'target_species': category_config.get('species', ['Mixed'])[:3],
            'confidence': overall_confidence,
            'integration_method': 'multi_source_fusion'
        }
    
    def _query_pressure_trend_data(self, station_id):
        """Query Phase I data for pressure trend analysis"""
        try:
            db_manager = getattr(self, 'db_manager', None)
            if not db_manager:
                return None
                
            # Get pressure data for last 6 hours to calculate trend
            with db_manager.connection.cursor() as cursor:
                cursor.execute("""
                    SELECT barometric_pressure, observation_time
                    FROM ndbc_data 
                    WHERE station_id = ? AND barometric_pressure IS NOT NULL
                    AND observation_time > ?
                    ORDER BY observation_time DESC 
                    LIMIT 12
                """, (station_id, int(time.time()) - 21600))  # Last 6 hours
                
                results = cursor.fetchall()
                if len(results) >= 2:
                    # Calculate trend
                    latest_pressure = results[0][0]
                    earliest_pressure = results[-1][0]
                    pressure_change = latest_pressure - earliest_pressure
                    
                    # Determine trend direction
                    if pressure_change > 0.05:
                        trend = 'rising'
                    elif pressure_change < -0.05:
                        trend = 'falling'
                    else:
                        trend = 'stable'
                    
                    return {
                        'barometric_pressure': latest_pressure,
                        'pressure_change_6h': pressure_change,
                        'pressure_trend': trend,
                        'observation_time': results[0][1],
                        'data_points': len(results)
                    }
                return None
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error querying pressure trend data for station {station_id}: {e}")
            return None
    
    def _query_tide_data_for_analysis(self, station_id):
        """Query Phase I tide data for fishing analysis"""
        try:
            db_manager = getattr(self, 'db_manager', None)
            if not db_manager:
                return None
                
            current_time = int(time.time())
            
            # Get current water level and next tide events
            with db_manager.connection.cursor() as cursor:
                # Current water level
                cursor.execute("""
                    SELECT water_level, observation_time
                    FROM coops_realtime 
                    WHERE station_id = ? AND water_level IS NOT NULL
                    ORDER BY observation_time DESC 
                    LIMIT 1
                """, (station_id,))
                
                current_level_result = cursor.fetchone()
                
                # Next tide events
                cursor.execute("""
                    SELECT tide_time, tide_height, tide_type
                    FROM tide_table 
                    WHERE station_id = ? AND tide_time > ?
                    ORDER BY tide_time ASC 
                    LIMIT 4
                """, (station_id, current_time))
                
                tide_events = cursor.fetchall()
                
                if current_level_result and tide_events:
                    # Determine current tide movement
                    next_tide = tide_events[0]
                    time_to_next_tide = next_tide[0] - current_time
                    
                    if time_to_next_tide < 3600:  # Within 1 hour of tide change
                        current_movement = 'slack'
                    elif next_tide[2] == 'H':  # Next is high tide
                        current_movement = 'incoming'
                    else:  # Next is low tide
                        current_movement = 'outgoing'
                    
                    return {
                        'current_water_level': current_level_result[0],
                        'current_movement': current_movement,
                        'next_tide_time': next_tide[0],
                        'next_tide_type': next_tide[2],
                        'time_to_next_tide_hours': time_to_next_tide / 3600,
                        'observation_time': current_level_result[1]
                    }
                return None
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error querying tide data for station {station_id}: {e}")
            return None
    
    def _analyze_pressure_trend_for_fishing(self, pressure_data, confidence):
        """Analyze pressure trend data for fishing implications"""
        pressure_trend = pressure_data.get('pressure_trend', 'stable')
        pressure_change = pressure_data.get('pressure_change_6h', 0.0)
        
        # Fishing quality based on pressure research (Skov et al. 2011)
        if pressure_trend == 'falling' and pressure_change < -0.1:
            quality = 'excellent'
            description = 'Rapidly falling pressure - prime fishing conditions'
        elif pressure_trend == 'falling':
            quality = 'good'
            description = 'Falling pressure - good fishing activity expected'
        elif pressure_trend == 'stable':
            quality = 'fair'
            description = 'Stable pressure - moderate fishing activity'
        else:  # rising
            quality = 'poor'
            description = 'Rising pressure - reduced fishing activity'
        
        return {
            'pressure_trend': pressure_trend,
            'pressure_change_6h': pressure_change,
            'quality': quality,
            'description': description,
            'confidence': confidence
        }
    
    def _analyze_tide_correlation_for_fishing(self, tide_data, quality_score, tide_source):
        """Analyze tide correlation data for fishing implications"""
        current_movement = tide_data.get('current_movement', 'unknown')
        time_to_next_tide = tide_data.get('time_to_next_tide_hours', 999)
        
        # Fishing quality based on tide movement
        if current_movement in ['incoming', 'outgoing']:
            if 1.0 <= time_to_next_tide <= 3.0:  # Sweet spot: 1-3 hours before tide change
                quality = 'excellent'
                description = f'Moving water ({current_movement}) - optimal fishing window'
            else:
                quality = 'good'
                description = f'Moving water ({current_movement}) - active fishing conditions'
        elif current_movement == 'slack':
            quality = 'fair'
            description = 'Slack tide - good for bottom fishing'
        else:
            quality = 'poor'
            description = 'Unknown tide conditions'
        
        return {
            'current_movement': current_movement,
            'time_to_next_tide_hours': time_to_next_tide,
            'quality': quality,
            'description': description,
            'quality_score': quality_score,
            'source_distance_miles': tide_source['distance_miles']
        }
    
    def _score_pressure_for_species(self, pressure_trend, pressure_change, preference):
        """Score pressure conditions for specific species preferences"""
        if preference == 'falling':
            if pressure_trend == 'falling':
                return 1.0 if pressure_change < -0.1 else 0.8
            elif pressure_trend == 'stable':
                return 0.5
            else:
                return 0.2
        elif preference == 'rising':
            if pressure_trend == 'rising':
                return 1.0 if pressure_change > 0.1 else 0.8
            elif pressure_trend == 'stable':
                return 0.5
            else:
                return 0.2
        else:  # stable preference
            if pressure_trend == 'stable':
                return 1.0
            else:
                return 0.6
    
    def _get_default_pressure_analysis(self):
        """Fallback pressure analysis when integration unavailable"""
        return {
            'pressure_trend': 'stable',
            'pressure_change_6h': 0.0,
            'quality': 'fair',
            'description': 'Default pressure conditions - integration unavailable',
            'confidence': 0.5
        }
    
    def _get_default_tide_analysis(self):
        """Fallback tide analysis when integration unavailable"""
        return {
            'current_movement': 'unknown',
            'time_to_next_tide_hours': 999,
            'quality': 'fair',
            'description': 'Default tide conditions - integration unavailable',
            'quality_score': 0.5
        }
    

class FishingForecastSearchList(SearchList):
    """
    NEW CLASS: SearchList extension for WeeWX template integration
    Provides fishing forecast data to WeeWX templates
    """
    
    def __init__(self, generator):
        super(FishingForecastSearchList, self).__init__(generator)
        
    def get_extension_list(self, timespan, db_lookup):
        """Return search list with fishing forecast data for templates"""
        
        try:
            # Get database manager
            db_manager = db_lookup()
            
            # Get all active fishing spots
            fishing_spots = self._get_active_fishing_spots(db_manager)
            
            # Build forecast data for each spot
            fishing_forecasts = {}
            for spot in fishing_spots:
                spot_data = self._get_spot_forecast_data(spot, db_manager)
                if spot_data:
                    fishing_forecasts[spot['name']] = spot_data
            
            # Get overall summary
            summary_data = self._get_overall_fishing_summary(fishing_forecasts)
            
            return [{
                'fishing_forecasts': fishing_forecasts,
                'fishing_summary': summary_data,
                'fishing_spots_count': len(fishing_spots),
                'last_fishing_update': self._get_last_update_time(db_manager)
            }]
            
        except Exception as e:
            log.error(f"Error in FishingForecastSearchList: {e}")
            return [{}]
    
    def _get_active_fishing_spots(self, db_manager):
        """Get all active fishing spots from CONF configuration for SearchList"""
        
        spots = []
        
        try:
            # Access config through generator
            config_dict = self.generator.config_dict
            service_config = config_dict.get('SurfFishingService', {})
            fishing_spots_config = service_config.get('fishing_spots', {})
            
            for spot_id, spot_config in fishing_spots_config.items():
                # Check if spot is active
                is_active = spot_config.get('active', 'true').lower() in ['true', '1', 'yes']
                
                if is_active:
                    spot = {
                        'id': spot_id,
                        'name': spot_config.get('name', spot_id),
                        'latitude': float(spot_config.get('latitude', 0.0)),
                        'longitude': float(spot_config.get('longitude', 0.0)),
                        'location_type': spot_config.get('location_type', 'shore'),
                        'target_category': spot_config.get('target_category', 'mixed_bag')
                    }
                    spots.append(spot)
            
            log.debug(f"SearchList loaded {len(spots)} fishing spots from CONF")
            
        except Exception as e:
            log.error(f"SearchList error getting fishing spots from CONF: {e}")
        
        return spots
    
    def _get_spot_forecast_data(self, spot, db_manager):
        """Get complete forecast data for a specific fishing spot"""
        
        try:
            # Create FishingForecastGenerator instance to use its methods
            generator = FishingForecastGenerator(self.generator.config_dict)
            
            return {
                'spot_info': spot,
                'current_forecast': generator.get_current_fishing_forecast(spot['id'], db_manager),
                'next_good_period': generator.find_next_good_fishing_period(spot['id'], db_manager),
                'today_summary': generator.get_today_fishing_summary(spot['id'], db_manager)
            }
            
        except Exception as e:
            log.error(f"Error getting fishing forecast data for {spot['name']}: {e}")
            return None
    
    def _get_overall_fishing_summary(self, fishing_forecasts):
        """Generate overall fishing summary across all spots"""
        
        if not fishing_forecasts:
            return {'status': 'No fishing spots configured'}
        
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
        
        return {
            'status': status,
            'best_rating': best_rating,
            'best_spot': best_spot,
            'total_spots': len(fishing_forecasts)
        }
    
    def _get_last_update_time(self, db_manager):
        """Get timestamp of last forecast update"""
        
        with db_manager.connection.cursor() as cursor:
            cursor.execute("SELECT MAX(generated_time) FROM marine_forecast_fishing_data")
            result = cursor.fetchone()
            
            if result and result[0]:
                return datetime.fromtimestamp(result[0]).strftime('%m/%d %I:%M %p')
            
            return 'Never'


class SurfFishingService(StdService):
    """
    Main service class for surf and fishing forecasts
    Follows WeeWX 5.1 service patterns exactly
    """
    
    def __init__(self, engine, config_dict):
        """
        Initialize SurfFishingService using WeeWX patterns
        All configuration from CONF only - no YAML access
        """
        super(SurfFishingService, self).__init__(engine, config_dict)
        
        # Store references for proper WeeWX integration
        self.engine = engine
        self.config_dict = config_dict
        
        # Get database manager from engine (WeeWX 5.1 pattern)
        self.db_manager = self.engine.db_binder.get_manager('wx_binding')
        
        # Read service configuration from CONF only
        self.service_config = config_dict.get('SurfFishingService', {})
        
        # Initialize GRIB processor
        self.grib_processor = GRIBProcessor(config_dict)
        if self.grib_processor.is_available():
            log.info("Using pygrib for GRIB processing")
        else:
            log.warning("No GRIB library available - WaveWatch III forecasts disabled")
        
        # Initialize forecast generators with CONF-based config
        self.surf_generator = SurfForecastGenerator(config_dict, self.db_manager)
        self.fishing_generator = FishingForecastGenerator(config_dict, self.db_manager)
        
        # Set up forecast timing from CONF
        self.forecast_interval = int(self.service_config.get('forecast_interval', 21600))  # 6 hours default
        self.shutdown_event = threading.Event()
        
        # Initialize station integration with CONF-based error handling
        try:
            log.info(f"{CORE_ICONS['status']} Initializing marine station integration...")
            
            # Load field definitions from CONF (written by installer)
            field_definitions = self._load_field_definitions()
            
            # Initialize integration components if field definitions available
            if field_definitions:
                log.info(f"{CORE_ICONS['status']} Station integration initialized successfully")
            else:
                log.warning(f"{CORE_ICONS['warning']} Station integration unavailable - using defaults")
                
        except Exception as e:
            log.error(f"{CORE_ICONS['warning']} Error initializing station integration: {e}")
            log.warning(f"{CORE_ICONS['warning']} Station integration unavailable - using defaults")
        
        # Start forecast generation
        log.info("Starting forecast generation loop")
        self._start_forecast_thread()
        
        log.info("Generating forecasts for all locations")
        log.info("SurfFishingService initialized successfully")
    
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
        """Start the background forecast generation thread"""
        
        self.forecast_thread = threading.Thread(
            target=self._forecast_loop,
            name="SurfFishingForecast",
            daemon=True
        )
        self.forecast_thread.start()
        
        log.info("Forecast generation thread started")
    
    def _forecast_loop(self):
        """Main forecast generation loop"""
        
        log.info("Starting forecast generation loop")
        
        while not self.shutdown_event.is_set():
            try:
                # Generate forecasts for all active locations
                self._generate_all_forecasts()
                
                # Wait for next forecast interval or shutdown signal
                self.shutdown_event.wait(timeout=self.forecast_interval)
                
            except Exception as e:
                log.error(f"Error in forecast loop: {e}")
                # Wait before retrying
                self.shutdown_event.wait(timeout=300)  # 5 minutes
    
    def _generate_all_forecasts(self):
        """Generate forecasts for all active surf and fishing spots"""
        
        log.info("Generating forecasts for all locations")
        
        try:
            # Get active surf spots
            surf_spots = self._get_active_surf_spots()
            
            # Get active fishing spots
            fishing_spots = self._get_active_fishing_spots()
            
            # Generate surf forecasts
            for spot in surf_spots:
                try:
                    self._generate_surf_forecast_for_spot(spot)
                except Exception as e:
                    log.error(f"Error generating surf forecast for {spot['name']}: {e}")
            
            # Generate fishing forecasts
            for spot in fishing_spots:
                try:
                    self._generate_fishing_forecast_for_spot(spot)
                except Exception as e:
                    log.error(f"Error generating fishing forecast for {spot['name']}: {e}")
            
            log.info(f"Forecast generation completed for {len(surf_spots)} surf spots and {len(fishing_spots)} fishing spots")
        
        except Exception as e:
            log.error(f"Error in forecast generation: {e}")
    
    def _get_active_surf_spots(self):
        """Get all active surf spots from CONF configuration"""
        
        spots = []
        
        try:
            # Read from CONF instead of database
            service_config = self.config_dict.get('SurfFishingService', {})
            surf_spots_config = service_config.get('surf_spots', {})
            
            for spot_id, spot_config in surf_spots_config.items():
                # Check if spot is active (default to True if not specified)
                is_active = spot_config.get('active', 'true').lower() in ['true', '1', 'yes']
                
                if is_active:
                    # Convert CONF data to expected format
                    spot = {
                        'id': spot_id,  # Use CONF key as ID
                        'name': spot_config.get('name', spot_id),
                        'latitude': float(spot_config.get('latitude', 0.0)),
                        'longitude': float(spot_config.get('longitude', 0.0)),
                        'bottom_type': spot_config.get('bottom_type', 'sand'),
                        'exposure': spot_config.get('exposure', 'exposed'),
                        'type': spot_config.get('type', 'surf')
                    }
                    spots.append(spot)
                    
            log.debug(f"Loaded {len(spots)} active surf spots from CONF")
            
        except Exception as e:
            log.error(f"Error getting surf spots from CONF: {e}")
            log.warning("Using fallback empty surf spots list")
        
        return spots

    def _get_active_fishing_spots(self):
        """Get all active fishing spots from CONF configuration"""
        
        spots = []
        
        try:
            # Read from CONF instead of database
            service_config = self.config_dict.get('SurfFishingService', {})
            fishing_spots_config = service_config.get('fishing_spots', {})
            
            for spot_id, spot_config in fishing_spots_config.items():
                # Check if spot is active (default to True if not specified)
                is_active = spot_config.get('active', 'true').lower() in ['true', '1', 'yes']
                
                if is_active:
                    # Convert CONF data to expected format
                    spot = {
                        'id': spot_id,  # Use CONF key as ID
                        'name': spot_config.get('name', spot_id),
                        'latitude': float(spot_config.get('latitude', 0.0)),
                        'longitude': float(spot_config.get('longitude', 0.0)),
                        'location_type': spot_config.get('location_type', 'shore'),
                        'target_category': spot_config.get('target_category', 'mixed_bag'),
                        'type': spot_config.get('type', 'fishing')
                    }
                    spots.append(spot)
                    
            log.debug(f"Loaded {len(spots)} active fishing spots from CONF")
            
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
        Get current marine conditions from Phase I data using field mappings from CONF
        Reads actual database field names and table names from Phase I configuration
        
        Args:
            latitude: Spot latitude (for future proximity-based selection)
            longitude: Spot longitude (for future proximity-based selection)
        
        Returns:
            Dictionary with available marine conditions
        
        Raises:
            RuntimeError: If Phase I field mappings are not available
        """
        
        # Get Phase I configuration from CONF
        phase_i_config = self.config_dict.get('MarineDataService', {})
        field_mappings = phase_i_config.get('field_mappings', {})
        
        # Get data source configuration from Phase II CONF
        service_config = self.config_dict.get('SurfFishingService', {})
        data_integration = service_config.get('data_integration', {})
        
        # Default conditions structure
        conditions = {
            'wave_height': None,
            'wave_period': None,
            'wind_speed': None,
            'wind_direction': None,
            'barometric_pressure': None,
            'water_level': None,
            'water_temperature': None,
            'data_quality': 'limited',
            'data_age_hours': 999
        }
        
        try:
            # Use existing method signature (latitude, longitude)
            
            # Use WeeWX database manager with proper error handling
            if not hasattr(self, 'db_manager') or self.db_manager is None:
                log.error("Database manager not available")
                return conditions
            
            # Check if Phase I field mappings are available - REQUIRED for Phase II
            if not field_mappings:
                error_msg = "Phase I Marine Data Extension field mappings not found in CONF. Phase II requires Phase I to be properly installed and configured."
                log.error(f"{CORE_ICONS['warning']} {error_msg}")
                raise RuntimeError(error_msg)
            
            # Get recent data time threshold from CONF
            max_age_hours = int(data_integration.get('max_data_age_hours', 6))
            recent_time = int(time.time()) - (max_age_hours * 3600)
            
            try:
                # Use connection for proper connection handling
                connection = self.db_manager.connection
                cursor = connection.cursor()
                
                # Get NDBC data using Phase I field mappings
                ndbc_fields = field_mappings.get('ndbc_module', {})
                if ndbc_fields:
                    ndbc_query_fields = []
                    ndbc_field_map = {}
                    
                    # Build dynamic query based on Phase I field mappings
                    for logical_name, field_config in ndbc_fields.items():
                        if isinstance(field_config, dict) and 'database_field' in field_config:
                            db_field = field_config['database_field']
                            ndbc_query_fields.append(db_field)
                            ndbc_field_map[db_field] = logical_name
                    
                    if ndbc_query_fields:
                        # Add dateTime to query
                        ndbc_query_fields.append('dateTime')
                        
                        # Get table name from first field (they should all be same table)
                        table_name = list(ndbc_fields.values())[0].get('database_table', 'ndbc_data')
                        
                        try:
                            query = f"""
                                SELECT {', '.join(ndbc_query_fields)}
                                FROM {table_name}
                                WHERE dateTime > ? 
                                ORDER BY dateTime DESC 
                                LIMIT 1
                            """
                            cursor.execute(query, (recent_time,))
                            
                            ndbc_result = cursor.fetchone()
                            
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
                            
                        except Exception as e:
                            log.warning(f"Could not get Phase I NDBC data: {e}")
                
                # Get CO-OPS data using Phase I field mappings
                coops_fields = field_mappings.get('coops_module', {})
                if coops_fields:
                    coops_query_fields = []
                    coops_field_map = {}
                    
                    # Build dynamic query based on Phase I field mappings
                    for logical_name, field_config in coops_fields.items():
                        if isinstance(field_config, dict) and 'database_field' in field_config:
                            db_field = field_config['database_field']
                            coops_query_fields.append(db_field)
                            coops_field_map[db_field] = logical_name
                    
                    if coops_query_fields:
                        # Add dateTime to query
                        coops_query_fields.append('dateTime')
                        
                        # Get table name from first field
                        table_name = list(coops_fields.values())[0].get('database_table', 'coops_realtime')
                        
                        try:
                            query = f"""
                                SELECT {', '.join(coops_query_fields)}
                                FROM {table_name}
                                WHERE dateTime > ? 
                                ORDER BY dateTime DESC 
                                LIMIT 1
                            """
                            cursor.execute(query, (recent_time,))
                            
                            coops_result = cursor.fetchone()
                            
                            if coops_result and len(coops_result) > 0:
                                data_age = (time.time() - coops_result[-1]) / 3600  # dateTime is last field
                                
                                # Map database results to logical names using Phase I mappings
                                for i, db_field in enumerate(coops_query_fields[:-1]):  # Skip dateTime
                                    if coops_result[i] is not None:
                                        logical_name = coops_field_map.get(db_field)
                                        if logical_name == 'current_water_level':
                                            conditions['water_level'] = coops_result[i]
                                        elif logical_name in ['coastal_water_temp', 'water_temperature']:
                                            if conditions['water_temperature'] is None:
                                                conditions['water_temperature'] = coops_result[i]
                                
                                # Update data quality based on freshest data
                                if conditions['data_age_hours'] > data_age:
                                    conditions['data_age_hours'] = data_age
                                    conditions['data_quality'] = 'good' if data_age < 1 else 'fair'
                                
                                log.debug(f"Retrieved CO-OPS data from {data_age:.1f} hours ago using Phase I mappings")
                            else:
                                log.debug("No recent CO-OPS data available")
                            
                        except Exception as e:
                            log.warning(f"Could not get Phase I CO-OPS data: {e}")
                
                cursor.close()
                
            except Exception as e:
                log.error(f"Database connection error: {e}")
            
            # Log data quality summary
            available_fields = sum(1 for k, v in conditions.items() 
                                if k not in ['data_quality', 'data_age_hours'] and v is not None)
            log.debug(f"Marine conditions: {available_fields} fields available, quality: {conditions['data_quality']}")
            
            return conditions
            
        except Exception as e:
            log.error(f"Error getting marine conditions: {e}")
            return conditions
    
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
                    'latitude': float(spot_config.get('latitude', 0.0)),
                    'longitude': float(spot_config.get('longitude', 0.0)),
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
                    lat = float(spot_config.get('latitude', 0))
                    lon = float(spot_config.get('longitude', 0))
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
                    lat = float(spot_config.get('latitude', 0))
                    lon = float(spot_config.get('longitude', 0))
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