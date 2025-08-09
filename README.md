# WeeWX Surf & Fishing Forecast Extension

**Local surf and fishing forecast system for your WeeWX weather station**

Phase II: Local Surf & Fishing Forecast System  
Magic Animal: Seahorse 🐟

## What This Extension Provides

### 🏄‍♂️ Surf Forecasting
- **Hourly surf conditions** - Wave height, period, direction with quality ratings
- **Wind quality assessment** - Offshore/onshore/cross wind impact on surf quality  
- **Spot-specific forecasts** - Customized for your local surf breaks
- **1-5 star rating system** - Easy-to-understand surf quality ratings
- **72-hour forecasts** - 3-day surf outlook for planning sessions

### 🎣 Fishing Forecasts
- **Period-based predictions** - 6 fishing periods per day (early morning through night)
- **Barometric pressure analysis** - Fish activity based on pressure trends
- **Tide movement correlation** - Optimal fishing times based on tide changes
- **Species-specific forecasts** - Targeted predictions for your preferred fish species
- **Activity ratings** - 1-5 star system for fishing conditions

### 🌊 Data Integration
- **Phase I Dependency** - Builds on existing marine data from Phase I extension
- **WaveWatch III Forecasts** - NOAA offshore wave model integration via GRIB files
- **Station Supplementation** - Optional use of your WeeWX station for local conditions
- **Multi-source Fusion** - Combines buoy, station, and model data intelligently

## Prerequisites

### Required
- **WeeWX 5.1+** - Check with `weectl --version`
- **Phase I Marine Data Extension** - Must be installed and operational first
- **Python 3.7+** - Check with `python3 --version`
- **Internet connection** - Required for WaveWatch III GRIB downloads

### Recommended
- **GRIB Processing Library** - For WaveWatch III forecasts:
  - `eccodes-python` (preferred) - Installed automatically via APT
  - `pygrib` (fallback) - Alternative GRIB processor

## Installation

### 1. Install Phase I Dependency
This extension requires the Phase I Marine Data Extension to be installed first:
```bash
# Install Phase I if not already installed
sudo weectl extension install weewx-marine-data-1.0.0-alpha.zip
sudo systemctl restart weewx
```

### 2. Download Phase II Extension
```bash
wget https://github.com/inguy24/weewx-fish_and_surf_forecasts/releases/download/v1.0.0-alpha/weewx-surf-fishing-1.0.0-alpha.zip
```

### 3. Install Phase II Extension
```bash
sudo weectl extension install weewx-surf-fishing-1.0.0-alpha.zip
```

### 4. Interactive Configuration
The installer will guide you through:

#### GRIB Library Installation
- Automatic detection and installation of GRIB processing libraries
- Fallback options if primary installation fails
- WaveWatch III capability validation

#### Atmospheric Data Strategy
Choose your approach for wind and pressure data:

1. **NOAA APIs Only** (recommended for most users)
   - Uses NDBC buoys, CO-OPS tides, WaveWatch III
   - Most comprehensive offshore data coverage

2. **WeeWX Station + NOAA Supplement** 
   - Uses your weather station for local conditions
   - Supplements with NOAA marine data
   - Better local wind/pressure accuracy within 5-10 miles

#### Location Configuration
Configure up to 10 total locations (surf + fishing combined):

**Surf Spots:**
- Name and coordinates
- Bottom type (sand, reef, point, jetty)
- Exposure level (exposed, semi-protected, protected)
- Maximum 5 surf spots

**Fishing Spots:**
- Name and coordinates  
- Location type (shore, pier, boat, mixed)
- Target species category
- Maximum 5 fishing spots

### 5. Restart WeeWX
```bash
sudo systemctl restart weewx
```

## Configuration Details

### Database Tables Created
- `marine_forecast_surf_spots` - Your surf spot locations and characteristics
- `marine_forecast_fishing_spots` - Your fishing spot locations and target species
- `marine_forecast_surf_data` - Current surf forecasts (replaced every 6 hours)
- `marine_forecast_fishing_data` - Current fishing forecasts (replaced every 6 hours)

### Forecast Generation
- **Update Frequency:** Every 6 hours (matching NOAA model runs)
- **Forecast Range:** 72 hours (3 days)
- **Surf Periods:** 8 per day (every 3 hours)
- **Fishing Periods:** 6 per day (4-hour periods)

### WeeWX Template Integration
The extension provides SearchList classes for template access:

#### Surf Data Access
```html
<!-- Access surf forecast data in templates -->
#for $spot in $surf_spots
<h3>$spot.name</h3>
<p>Current: $spot.current_conditions.conditions_text ($spot.current_conditions.rating_stars)</p>
<p>Wave Height: $spot.current_conditions.wave_height_range</p>
<p>Next Good Session: $spot.next_good_session.time ($spot.next_good_session.rating stars)</p>
#end for
```

#### Fishing Data Access  
```html
<!-- Access fishing forecast data in templates -->
#for $spot in $fishing_spots
<h3>$spot.name ($spot.target_category)</h3>
<p>Today's Best: $spot.today_summary.best_period ($spot.today_summary.best_rating stars)</p>
<p>Next Good Period: $spot.next_good_period.period - $spot.next_good_period.conditions</p>
#end for
```

## Data Sources and Coverage

### Phase I Integration (Required)
Reads data from Phase I marine extension tables:
- **NDBC buoy data** - Wave height, period, direction, wind, pressure
- **CO-OPS tide stations** - Current water levels and coastal water temperature
- **Tide predictions** - 7-day rolling high/low tide forecasts

### WaveWatch III Integration (New)
- **Global wave model** - NOAA's premier wave forecasting system
- **Regional grids** - Higher resolution for US coasts (7km) and Great Lakes (2.5km)
- **GRIB2 format** - Industry-standard meteorological data format
- **72-hour forecasts** - Offshore wave conditions before local transformation

### Station Integration (Optional)
Your WeeWX weather station can supplement marine data:
- **Wind data** - More accurate local wind conditions within 5 miles
- **Pressure data** - Local barometric pressure trends within 10 miles
- **Smart fallbacks** - Automatic selection of best available data source

## Forecast Algorithms

### Surf Quality Assessment
Scientific approach based on established surf forecasting principles:

1. **Wave Period Analysis**
   - 12+ seconds: Excellent (ground swell)
   - 8-12 seconds: Good quality
   - 6-8 seconds: Fair conditions  
   - <6 seconds: Poor (wind swell)

2. **Wind Quality Factor**
   - Offshore winds: +2 rating (clean, groomed waves)
   - Cross-shore winds: No change (variable conditions)
   - Onshore winds: -2 rating (choppy, messy conditions)
   - Calm conditions: +1 rating (glassy surface)

3. **Local Transformation**
   - Bottom type effects (reef focuses energy, sand dissipates)
   - Exposure factors (protected areas see reduced swell)
   - Spot-specific characteristics

### Fishing Activity Prediction
Based on established fishing pressure and biological patterns:

1. **Barometric Pressure**
   - Falling pressure: Fish very active (best conditions)
   - Stable pressure: Normal activity levels
   - Rising pressure: Fish less active

2. **Tide Movement**
   - Moving water: Active feeding (2 hours before/after tide change)
   - Slack tide: Minimal activity (high/low tide ±30 minutes)

3. **Time of Day**
   - Dawn/dusk: Prime feeding times
   - Early morning/late afternoon: Good activity
   - Midday: Fair (species dependent)
   - Night: Species dependent

4. **Species-Specific Factors**
   - Saltwater inshore: Tide-dependent, pressure-sensitive
   - Surf fishing: Wave height and tide movement critical
   - Pier fishing: Less location-sensitive, steady conditions

## Troubleshooting

### GRIB Processing Issues
```bash
# Check GRIB library installation
python3 -c "import eccodes; print('eccodes available')"
python3 -c "import pygrib; print('pygrib available')"

# Manual installation if needed
sudo apt-get install libeccodes0 libeccodes-dev python3-eccodes
```

### Phase I Dependency Issues
```bash
# Verify Phase I is installed and running
sudo tail -f /var/log/syslog | grep MarineDataService

# Check Phase I database tables exist
weectl database query "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'ndbc%'"
```

### Forecast Generation Issues
```bash
# Check service status
sudo tail -f /var/log/syslog | grep SurfFishingService

# Verify database tables created
weectl database query "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'marine_forecast%'"

# Check for recent forecasts
weectl database query "SELECT COUNT(*) FROM marine_forecast_surf_data WHERE generated_time > strftime('%s', 'now', '-1 day')"
```

### Performance Optimization
- **GRIB Cache Size**: ~200MB peak storage for temporary GRIB files
- **Database Size**: ~210KB per 10 locations (7-day retention)
- **Update Frequency**: Configurable (default 6 hours)
- **Cleanup**: Automatic removal of old forecasts and temporary files

## Configuration Examples

### weewx.conf Section
```ini
[SurfFishingService]
    enable = true
    forecast_interval = 21600
    
    [[station_integration]]
        type = station_supplement
        
        [[[sensors]]]
            wind = true
            pressure = true
            temperature = false
    
    [[grib_processing]]
        available = true
        library = eccodes
    
    [[surf_spots]]
        [[[spot_0]]]
            name = Malibu
            latitude = 34.0259
            longitude = -118.7798
            bottom_type = sand
            exposure = exposed
    
    [[fishing_spots]]
        [[[spot_0]]]
            name = Santa Monica Pier
            latitude = 34.0089
            longitude = -118.4973
            location_type = pier
            target_category = mixed_bag
```

## Support and Development

### Bug Reports
- **GitHub Issues**: [Report bugs](https://github.com/inguy24/weewx-fish_and_surf_forecasts/issues)
- **Log Analysis**: Include relevant WeeWX log entries
- **Configuration**: Share sanitized weewx.conf sections

### Feature Requests
- **Enhancement Issues**: [Request features](https://github.com/inguy24/weewx-fish_and_surf_forecasts/issues)
- **Community Discussion**: Share ideas and improvements
- **Algorithm Suggestions**: Improvements to surf/fishing prediction models

### Documentation
- **Project Wiki**: [Detailed documentation](https://github.com/inguy24/weewx-fish_and_surf_forecasts/wiki)
- **Template Examples**: Sample WeeWX template integrations
- **API Reference**: SearchList class documentation

### WeeWX Community
- **WeeWX User Group**: [General WeeWX support](https://groups.google.com/g/weewx-user)
- **Extension Discussion**: Marine forecasting topics

## Technical Details

### Architecture Overview
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Phase I       │    │  WaveWatch III   │    │  WeeWX Station  │
│  Marine Data    │    │   GRIB Files     │    │   (Optional)    │
│                 │    │                  │    │                 │
│ • NDBC buoys    │◄───┤ • Offshore waves │◄───┤ • Local wind    │
│ • CO-OPS tides  │    │ • 72hr forecasts │    │ • Local pressure│
│ • Tide preds    │    │ • Global coverage│    │ • Air temp      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                ┌────────────────▼────────────────┐
                │    Surf & Fishing Service       │
                │                                 │
                │ • Data fusion algorithms        │
                │ • Offshore→local transformation │
                │ • Quality assessment            │
                │ • Species activity prediction   │
                └────────────────┬────────────────┘
                                 │
                ┌────────────────▼────────────────┐
                │      Forecast Database          │
                │                                 │
                │ • Surf forecasts (hourly)       │
                │ • Fishing forecasts (periods)   │
                │ • User locations               │
                │ • Quality ratings              │
                └────────────────┬────────────────┘
                                 │
                ┌────────────────▼────────────────┐
                │    WeeWX Templates              │
                │                                 │
                │ • SearchList integration        │
                │ • Web interface display         │
                │ • Mobile-responsive design      │
                └─────────────────────────────────┘
```

### Data Flow Sequence
1. **Phase I Collection** - NDBC/CO-OPS data collected every 10 minutes to 1 hour
2. **WaveWatch III Download** - GRIB files downloaded every 6 hours
3. **Station Integration** - Local station data processed in real-time
4. **Forecast Generation** - Algorithms run every 6 hours for all locations
5. **Database Storage** - Current forecasts replace old data (no historical storage)
6. **Template Access** - SearchList provides data to WeeWX templates

### Algorithm Validation
The forecast algorithms are based on established meteorological and biological principles:

#### Surf Forecasting Science
- **Dispersion Theory**: Deep water wave period determines quality
- **Shoaling Effects**: Wave height transformation in shallow water
- **Wind-Wave Interaction**: Surface wind effects on wave quality
- **Local Bathymetry**: Bottom type and exposure effects

#### Fishing Activity Science  
- **Barometric Sensitivity**: Fish behavior response to pressure changes
- **Circadian Rhythms**: Dawn/dusk feeding patterns in most species
- **Tidal Influence**: Water movement triggering feeding behavior
- **Species Ecology**: Different species have different optimal conditions

### Performance Characteristics
- **Startup Time**: ~30 seconds (includes GRIB library detection)
- **Forecast Generation**: ~30 seconds per location
- **Memory Usage**: ~50MB base + 200MB peak during GRIB processing
- **Disk Usage**: ~210KB per 10 locations (7-day retention)
- **Network Usage**: ~50MB per 6-hour forecast update (GRIB downloads)

## Comparison with Commercial Services

### Advantages
- **Local Integration**: Uses your exact weather station data
- **Customizable**: Spot-specific characteristics and preferences
- **No Subscription**: Free access to NOAA data sources
- **Privacy**: All data processing happens locally
- **Extensible**: Open source, customizable algorithms

### Commercial Comparison
| Feature | This Extension | Surfline | MagicSeaweed | Fishbrain |
|---------|---------------|----------|--------------|-----------|
| Cost | Free | $5-10/month | $3-8/month | $8-12/month |
| Local Station Data | ✅ | ❌ | ❌ | ❌ |
| Custom Spots | ✅ | Limited | Limited | ✅ |
| Fishing Forecasts | ✅ | ❌ | ❌ | ✅ |
| Offline Access | ✅ | ❌ | ❌ | ❌ |
| Algorithm Transparency | ✅ | ❌ | ❌ | ❌ |

## License and Credits

### License
This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

### Copyright
Copyright (C) 2025 Shane Burkhardt - [GitHub Profile](https://github.com/inguy24)

### Data Sources
- **NOAA WaveWatch III** - Global wave forecasting model
- **NOAA NDBC** - Buoy observations (via Phase I)
- **NOAA CO-OPS** - Tide and water level data (via Phase I)
- **WeeWX Community** - Software framework and development guidance

### Acknowledgments
- **WeeWX Development Team** - Excellent weather station software platform
- **NOAA** - Comprehensive free marine data APIs
- **Phase I Extension** - Foundation marine data collection
- **Marine Meteorology Community** - Scientific validation and feedback
- **Beta Testers** - Real-world validation and bug reports

### Citations
Algorithms based on established scientific literature:
- Coastal wave transformation theory (Komar, 1998)
- Fish behavior and barometric pressure (Skov et al., 2011)
- Surf forecasting principles (Surfline/CDIP methodologies)
- Tidal influence on marine species (Gibson, 2003)

## Version History

### v1.0.0-alpha (Current)
- Initial release with Phase A core infrastructure
- GRIB processing capability (eccodes-python/pygrib)
- Multi-source atmospheric data integration
- Basic surf and fishing forecast algorithms
- WeeWX 5.1 database integration
- SearchList template integration

### Planned Updates
- **v1.1.0** - Enhanced algorithms with historical validation
- **v1.2.0** - Advanced web interface with charts
- **v2.0.0** - Mobile app integration and notifications
- **v2.1.0** - Machine learning forecast improvements

---

**Current Version**: 1.0.0-alpha  
**WeeWX Compatibility**: 5.1+  
**License**: GPL v3.0  
**Magic Animal**: Seahorse 🐟

For installation support, bug reports, or feature requests, please visit our [GitHub repository](https://github.com/inguy24/weewx-fish_and_surf_forecasts).