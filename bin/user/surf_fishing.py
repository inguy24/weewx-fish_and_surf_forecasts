#!/usr/bin/env python3
# Magic Animal: Seahorse 🐟
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
from typing import Dict, List, Optional, Any

# WeeWX imports
import weewx
import weewx.manager
from weewx.engine import StdService
import weeutil.logger

# Logging setup
log = weeutil.logger.logging.getLogger(__name__)

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
    
    def __init__(self, config_dict):
        self.config_dict = config_dict
        service_config = config_dict.get('SurfFishingService', {})
        self.surf_rating_factors = service_config.get('surf_rating_factors', {})
    
    def generate_surf_forecast(self, spot_config, marine_conditions, wavewatch_data):
        """Generate surf forecast for a specific spot"""
        
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
    
    def _assess_surf_quality(self, surf_forecast, current_wind, spot_config):
        """Assess surf quality based on wind and wave conditions"""
        
        quality_forecast = []
        
        for period in surf_forecast:
            # Calculate wind condition relative to wave direction
            wind_condition = self._classify_wind_condition(
                period['wind_direction'], 
                period['wave_direction'],
                period['wind_speed']
            )
            
            # Calculate quality rating
            quality_rating = self._calculate_surf_rating(
                period['wave_period'], 
                wind_condition,
                (period['wave_height_min'] + period['wave_height_max']) / 2
            )
            
            # Add quality assessment
            enhanced_period = period.copy()
            enhanced_period.update({
                'wind_condition': wind_condition['type'],
                'quality_rating': quality_rating['rating'],
                'conditions_text': quality_rating['text'],
                'confidence': quality_rating['confidence']
            })
            
            quality_forecast.append(enhanced_period)
        
        return quality_forecast
    
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


class FishingForecastGenerator:
    """Generate fishing condition forecasts"""
    
    def __init__(self, config_dict):
        self.config_dict = config_dict
        service_config = config_dict.get('SurfFishingService', {})
        self.fish_categories = service_config.get('fish_categories', {})
        self.fishing_scoring = service_config.get('fishing_scoring', {})
    
    def generate_fishing_forecast(self, spot_config, marine_conditions):
        """Generate fishing forecast for a specific spot"""
        
        try:
            # Get target species category
            target_category = spot_config.get('target_category', 'mixed_bag')
            category_config = self.fish_categories.get(target_category, {})
            
            # Generate period-based forecasts (6 periods per day)
            forecast_periods = self._generate_fishing_periods()
            
            # Score each period
            scored_periods = []
            for period in forecast_periods:
                period_score = self._score_fishing_period(
                    period, marine_conditions, category_config
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
    
    def _score_fishing_period(self, period, marine_conditions, category_config):
        """Score a fishing period based on conditions"""
        
        total_score = 0
        scoring_factors = []
        
        # Pressure trend scoring (most important factor)
        pressure_score, pressure_factor = self._score_pressure_conditions(marine_conditions)
        total_score += pressure_score
        scoring_factors.append(pressure_factor)
        
        # Tide movement scoring
        tide_score, tide_factor = self._score_tide_conditions(period, marine_conditions)
        total_score += tide_score
        scoring_factors.append(tide_factor)
        
        # Time of day scoring
        time_score, time_factor = self._score_time_of_day(period)
        total_score += time_score
        scoring_factors.append(time_factor)
        
        # Species-specific adjustments
        species_score, species_factor = self._score_species_conditions(
            period, marine_conditions, category_config
        )
        total_score += species_score
        if species_factor:
            scoring_factors.append(species_factor)
        
        # Convert to 1-5 rating
        if total_score >= 6:
            rating = 5
            conditions_text = "Excellent"
        elif total_score >= 4.5:
            rating = 4
            conditions_text = "Good"
        elif total_score >= 3:
            rating = 3
            conditions_text = "Fair"
        elif total_score >= 1.5:
            rating = 2
            conditions_text = "Poor"
        else:
            rating = 1
            conditions_text = "Very Poor"
        
        # Determine species activity level
        if rating >= 4:
            species_activity = 'high'
        elif rating >= 3:
            species_activity = 'moderate'
        else:
            species_activity = 'low'
        
        # Enhanced period with scoring
        enhanced_period = period.copy()
        enhanced_period.update({
            'activity_rating': rating,
            'conditions_text': conditions_text,
            'species_activity': species_activity,
            'scoring_factors': scoring_factors,
            'total_score': total_score,
            'generated_time': int(time.time())
        })
        
        return enhanced_period
    
    def _score_pressure_conditions(self, marine_conditions):
        """Score barometric pressure conditions"""
        
        # Get pressure trend (this would come from analyzing recent pressure data)
        current_pressure = marine_conditions.get('current_pressure', 30.0)
        
        # Simplified pressure trend calculation
        # In production, this would analyze pressure change over last 3 hours
        pressure_change = 0  # Placeholder
        
        if pressure_change < -0.05:  # Falling fast
            score = 3
            factor = "Pressure falling rapidly - fish very active"
        elif pressure_change < -0.02:  # Falling slowly
            score = 2
            factor = "Pressure falling - increased fish activity"
        elif abs(pressure_change) <= 0.02:  # Stable
            score = 1
            factor = "Stable pressure - normal activity"
        else:  # Rising
            score = 0
            factor = "Rising pressure - fish less active"
        
        return score, factor
    
    def _score_tide_conditions(self, period, marine_conditions):
        """Score tide movement conditions"""
        
        # This would analyze tide table data for the period
        # Simplified for now
        
        # Assume moving water periods are best
        period_hour = period['period_start_hour']
        
        # Simplified tide scoring based on time (would use actual tide data)
        if period_hour in [6, 7, 18, 19]:  # Typical tide change times
            score = 2
            factor = "Moving water - active feeding"
        elif period_hour in [0, 12]:  # High/low tide times
            score = 0
            factor = "Slack tide - slow fishing"
        else:
            score = 1
            factor = "Moderate tide flow"
        
        return score, factor
    
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


class SurfFishingService(StdService):
    """
    Main service class for surf and fishing forecasts
    Follows WeeWX 5.1 service patterns exactly
    """
    
    def __init__(self, engine, config_dict):
        super(SurfFishingService, self).__init__(engine, config_dict)
        
        # Get database manager from engine (WeeWX 5.1 pattern)
        self.db_manager = self.engine.db_binder.get_manager('wx_binding')
        
        # Read configuration from weewx.conf
        self.service_config = config_dict.get('SurfFishingService', {})
        self.station_integration = self.service_config.get('station_integration', {})
        self.forecast_interval = int(self.service_config.get('forecast_interval', 21600))  # 6 hours
        
        # Initialize components
        self.grib_processor = GRIBProcessor(config_dict)
        self.wavewatch_collector = WaveWatchDataCollector(config_dict, self.grib_processor)
        self.surf_generator = SurfForecastGenerator(config_dict)
        self.fishing_generator = FishingForecastGenerator(config_dict)
        
        # Background forecast generation
        self.forecast_thread = None
        self.shutdown_event = threading.Event()
        
        # Start forecast generation thread
        self._start_forecast_thread()
        
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
        """Get all active surf spots from database"""
        
        spots = []
        
        try:
            with self.db_manager.connection as connection:
                cursor = connection.execute("""
                    SELECT id, name, latitude, longitude, bottom_type, exposure
                    FROM marine_forecast_surf_spots 
                    WHERE active = 1
                """)
                
                columns = [desc[0] for desc in cursor.description]
                for row in cursor.fetchall():
                    spot = dict(zip(columns, row))
                    spots.append(spot)
        
        except Exception as e:
            log.error(f"Error getting surf spots: {e}")
        
        return spots
    
    def _get_active_fishing_spots(self):
        """Get all active fishing spots from database"""
        
        spots = []
        
        try:
            with self.db_manager.connection as connection:
                cursor = connection.execute("""
                    SELECT id, name, latitude, longitude, location_type, target_category
                    FROM marine_forecast_fishing_spots 
                    WHERE active = 1
                """)
                
                columns = [desc[0] for desc in cursor.description]
                for row in cursor.fetchall():
                    spot = dict(zip(columns, row))
                    spots.append(spot)
        
        except Exception as e:
            log.error(f"Error getting fishing spots: {e}")
        
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
        """Get current marine conditions from Phase I data"""
        
        conditions = {}
        
        try:
            # Get latest NDBC data from Phase I
            with self.db_manager.connection as connection:
                cursor = connection.execute("""
                    SELECT marine_wave_height, marine_wave_period, marine_wave_direction,
                           marine_wind_speed, marine_wind_direction, marine_barometric_pressure,
                           marine_sea_surface_temp, dateTime
                    FROM ndbc_data 
                    WHERE dateTime > ? 
                    ORDER BY dateTime DESC LIMIT 1
                """, (int(time.time()) - 3600,))  # Last hour
                
                row = cursor.fetchone()
                if row:
                    columns = [desc[0] for desc in cursor.description]
                    ndbc_data = dict(zip(columns, row))
                    
                    conditions.update({
                        'current_wave_height': ndbc_data.get('marine_wave_height'),
                        'current_wave_period': ndbc_data.get('marine_wave_period'),
                        'current_wave_direction': ndbc_data.get('marine_wave_direction'),
                        'current_wind_speed': ndbc_data.get('marine_wind_speed'),
                        'current_wind_direction': ndbc_data.get('marine_wind_direction'),
                        'current_pressure': ndbc_data.get('marine_barometric_pressure'),
                        'current_sea_temp': ndbc_data.get('marine_sea_surface_temp')
                    })
        
        except Exception as e:
            log.warning(f"Could not get Phase I NDBC data: {e}")
        
        try:
            # Get latest CO-OPS data from Phase I
            with self.db_manager.connection as connection:
                cursor = connection.execute("""
                    SELECT marine_current_water_level, marine_coastal_water_temp, dateTime
                    FROM coops_realtime 
                    WHERE dateTime > ? 
                    ORDER BY dateTime DESC LIMIT 1
                """, (int(time.time()) - 1800,))  # Last 30 minutes
                
                row = cursor.fetchone()
                if row:
                    columns = [desc[0] for desc in cursor.description]
                    coops_data = dict(zip(columns, row))
                    
                    conditions.update({
                        'current_water_level': coops_data.get('marine_current_water_level'),
                        'current_coastal_temp': coops_data.get('marine_coastal_water_temp')
                    })
        
        except Exception as e:
            log.warning(f"Could not get Phase I CO-OPS data: {e}")
        
        try:
            # Get next tide predictions from Phase I
            with self.db_manager.connection as connection:
                cursor = connection.execute("""
                    SELECT tide_time, tide_type, predicted_height
                    FROM tide_table 
                    WHERE tide_time > ? 
                    ORDER BY tide_time ASC LIMIT 4
                """, (int(time.time()),))  # Future tides
                
                tides = cursor.fetchall()
                if tides:
                    # Find next high and low tides
                    for tide_time, tide_type, height in tides:
                        if tide_type == 'H' and 'next_high_tide' not in conditions:
                            conditions['next_high_tide'] = tide_time
                            conditions['next_high_height'] = height
                        elif tide_type == 'L' and 'next_low_tide' not in conditions:
                            conditions['next_low_tide'] = tide_time
                            conditions['next_low_height'] = height
        
        except Exception as e:
            log.warning(f"Could not get Phase I tide data: {e}")
        
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


class SurfForecastSearchList(weewx.cheetahgenerator.SearchList):
    """SearchList for WeeWX template integration - Surf Forecasts"""
    
    def __init__(self, generator):
        super(SurfForecastSearchList, self).__init__(generator)
    
    def get_extension_list(self, timespan, db_lookup):
        """Return surf forecast data for templates"""
        
        search_list = []
        
        try:
            # Get database manager
            with db_lookup().connection as connection:
                # Get all active surf spots
                spots_cursor = connection.execute("""
                    SELECT id, name, latitude, longitude, bottom_type, exposure
                    FROM marine_forecast_surf_spots 
                    WHERE active = 1
                    ORDER BY name
                """)
                
                surf_spots = []
                for row in spots_cursor.fetchall():
                    spot_id, name, lat, lon, bottom_type, exposure = row
                    
                    # Get current forecast for this spot
                    forecast_cursor = connection.execute("""
                        SELECT forecast_time, wave_height_min, wave_height_max, wave_period,
                               wind_condition, quality_rating, conditions_text, confidence
                        FROM marine_forecast_surf_data
                        WHERE spot_id = ? AND forecast_time > ?
                        ORDER BY forecast_time ASC
                        LIMIT 8
                    """, (spot_id, int(time.time())))
                    
                    forecast_data = []
                    for forecast_row in forecast_cursor.fetchall():
                        forecast_data.append({
                            'forecast_time': forecast_row[0],
                            'forecast_time_text': time.strftime('%m/%d %H:%M', time.localtime(forecast_row[0])),
                            'wave_height_min': forecast_row[1],
                            'wave_height_max': forecast_row[2],
                            'wave_height_range': f"{forecast_row[1]:.1f}-{forecast_row[2]:.1f} ft",
                            'wave_period': forecast_row[3],
                            'wind_condition': forecast_row[4],
                            'quality_rating': forecast_row[5],
                            'rating_stars': '★' * forecast_row[5] + '☆' * (5 - forecast_row[5]),
                            'conditions_text': forecast_row[6],
                            'confidence': forecast_row[7]
                        })
                    
                    # Get next best surf session
                    next_good_session = self._find_next_good_session(forecast_data)
                    
                    surf_spot = {
                        'id': spot_id,
                        'name': name,
                        'latitude': lat,
                        'longitude': lon,
                        'bottom_type': bottom_type,
                        'exposure': exposure,
                        'forecast': forecast_data,
                        'next_good_session': next_good_session,
                        'current_conditions': forecast_data[0] if forecast_data else None
                    }
                    surf_spots.append(surf_spot)
                
                search_list.append({'surf_spots': surf_spots})
        
        except Exception as e:
            log.error(f"Error generating surf forecast search list: {e}")
            search_list.append({'surf_spots': []})
        
        return search_list
    
    def _find_next_good_session(self, forecast_data):
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


class FishingForecastSearchList(weewx.cheetahgenerator.SearchList):
    """SearchList for WeeWX template integration - Fishing Forecasts"""
    
    def __init__(self, generator):
        super(FishingForecastSearchList, self).__init__(generator)
    
    def get_extension_list(self, timespan, db_lookup):
        """Return fishing forecast data for templates"""
        
        search_list = []
        
        try:
            # Get database manager
            with db_lookup().connection as connection:
                # Get all active fishing spots
                spots_cursor = connection.execute("""
                    SELECT id, name, latitude, longitude, location_type, target_category
                    FROM marine_forecast_fishing_spots 
                    WHERE active = 1
                    ORDER BY name
                """)
                
                fishing_spots = []
                for row in spots_cursor.fetchall():
                    spot_id, name, lat, lon, location_type, target_category = row
                    
                    # Get current forecast for this spot
                    forecast_cursor = connection.execute("""
                        SELECT forecast_date, period_name, period_start_hour, period_end_hour,
                               species_activity, activity_rating, conditions_text
                        FROM marine_forecast_fishing_data
                        WHERE spot_id = ? AND forecast_date >= ?
                        ORDER BY forecast_date, period_name
                        LIMIT 18
                    """, (spot_id, int(time.time()) - (int(time.time()) % 86400)))  # Today
                    
                    forecast_data = []
                    for forecast_row in forecast_cursor.fetchall():
                        forecast_data.append({
                            'forecast_date': forecast_row[0],
                            'forecast_date_text': time.strftime('%m/%d', time.localtime(forecast_row[0])),
                            'period_name': forecast_row[1],
                            'period_display': forecast_row[1].replace('_', ' ').title(),
                            'period_start_hour': forecast_row[2],
                            'period_end_hour': forecast_row[3],
                            'period_time_range': f"{forecast_row[2]:02d}:00-{forecast_row[3]:02d}:00",
                            'species_activity': forecast_row[4],
                            'activity_rating': forecast_row[5],
                            'rating_stars': '★' * forecast_row[5] + '☆' * (5 - forecast_row[5]),
                            'conditions_text': forecast_row[6]
                        })
                    
                    # Get next good fishing period
                    next_good_period = self._find_next_good_period(forecast_data)
                    
                    # Get today's summary
                    today_summary = self._get_today_fishing_summary(forecast_data)
                    
                    fishing_spot = {
                        'id': spot_id,
                        'name': name,
                        'latitude': lat,
                        'longitude': lon,
                        'location_type': location_type,
                        'target_category': target_category,
                        'forecast': forecast_data,
                        'next_good_period': next_good_period,
                        'today_summary': today_summary
                    }
                    fishing_spots.append(fishing_spot)
                
                search_list.append({'fishing_spots': fishing_spots})
        
        except Exception as e:
            log.error(f"Error generating fishing forecast search list: {e}")
            search_list.append({'fishing_spots': []})
        
        return search_list
    
    def _find_next_good_period(self, forecast_data):
        """Find next fishing period with rating >= 3"""
        
        current_time = int(time.time())
        
        for forecast in forecast_data:
            # Simple check - if it's today and rating is good
            if (forecast['forecast_date'] >= current_time - 86400 and 
                forecast['activity_rating'] >= 3):
                return {
                    'period': forecast['period_display'],
                    'date': forecast['forecast_date_text'],
                    'time_range': forecast['period_time_range'],
                    'rating': forecast['activity_rating'],
                    'conditions': forecast['conditions_text'],
                    'activity': forecast['species_activity']
                }
        return None
    
    def _get_today_fishing_summary(self, forecast_data):
        """Get today's fishing summary"""
        
        if not forecast_data:
            return {'status': 'No forecast available'}
        
        today = int(time.time()) - (int(time.time()) % 86400)
        today_forecasts = [f for f in forecast_data if f['forecast_date'] == today]
        
        if not today_forecasts:
            return {'status': 'No forecast for today'}
        
        # Find best period today
        best_period = max(today_forecasts, key=lambda x: x['activity_rating'])
        average_rating = sum(f['activity_rating'] for f in today_forecasts) / len(today_forecasts)
        
        return {
            'status': 'Available',
            'best_period': best_period['period_display'],
            'best_rating': best_period['activity_rating'],
            'best_conditions': best_period['conditions_text'],
            'average_rating': round(average_rating, 1),
            'total_periods': len(today_forecasts)
        }


# Test function for development/debugging
def test_forecast_generation():
    """Test forecast generation functionality"""
    
    print("Testing Surf & Fishing Forecast Generation")
    print("Magic Animal: Seahorse 🐟")
    print("-" * 50)
    
    # Test GRIB processor
    config_dict = {}
    grib_processor = GRIBProcessor(config_dict)
    print(f"GRIB processing available: {grib_processor.is_available()}")
    print(f"GRIB library: {grib_processor.grib_library}")
    
    # Test forecast generators
    surf_generator = SurfForecastGenerator(config_dict)
    fishing_generator = FishingForecastGenerator(config_dict)
    
    print("Forecast generators initialized successfully")
    
    # Mock data for testing
    test_spot = {
        'name': 'Test Beach',
        'latitude': 34.0522,
        'longitude': -118.2437,
        'bottom_type': 'sand',
        'exposure': 'exposed'
    }
    
    test_marine_conditions = {
        'current_wave_height': 3.5,
        'current_wave_period': 8.0,
        'current_wind_speed': 10.0,
        'current_pressure': 30.15
    }
    
    # Test surf forecast
    print("\nTesting surf forecast generation...")
    surf_forecast = surf_generator.generate_surf_forecast(
        test_spot, test_marine_conditions, []
    )
    print(f"Generated {len(surf_forecast)} surf forecast periods")
    
    # Test fishing forecast
    print("\nTesting fishing forecast generation...")
    fishing_forecast = fishing_generator.generate_fishing_forecast(
        test_spot, test_marine_conditions
    )
    print(f"Generated {len(fishing_forecast)} fishing forecast periods")
    
    print("\nTest completed successfully!")


if __name__ == "__main__":
    # Run tests if executed directly
    test_forecast_generation()_i_marine_conditions(spot['latitude'], spot['longitude'])
        
        # Get WaveWatch III forecast data
        wavewatch_data = self.wavewatch_collector.fetch_forecast_data(
            spot['latitude'], spot['longitude']
        )
        
        # Generate surf forecast
        surf_forecast = self.surf_generator.generate_surf_forecast(
            spot, marine_conditions, wavewatch_data
        )
        
        # Store forecast in database
        self._store_surf_forecast(spot['id'], surf_forecast)
    
    def _generate_fishing_forecast_for_spot(self, spot):
        """Generate fishing forecast for a specific spot"""
        
        log.debug(f"Generating fishing forecast for {spot['name']}")
        
        # Get current marine conditions from Phase I
        marine_conditions = self._get_phase