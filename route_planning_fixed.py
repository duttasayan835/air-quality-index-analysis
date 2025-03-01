# Travel mode selection with icons
travel_modes = {
    "Walking 🚶": "Walking",
    "Cycling 🚲": "Cycling",
    "Driving 🚗": "Driving"
}

selected_mode_key = st.radio(
    "Travel Mode",
    options=list(travel_modes.keys()),
    horizontal=True
)
selected_mode = travel_modes[selected_mode_key]

if st.button("Find Green Routes", use_container_width=True):
    if start_point and destination:
        with st.spinner("Calculating route and environmental metrics..."):
            # Get coordinates
            start_coords = get_coordinates(start_point)
            end_coords = get_coordinates(destination)
            
            if not start_coords:
                st.error(f"Could not find coordinates for '{start_point}'. Please check the spelling and try again.")
            elif not end_coords:
                st.error(f"Could not find coordinates for '{destination}'. Please check the spelling and try again.")
            else:
                # Get route details
                route_data = get_route(start_coords, end_coords, selected_mode)
                
                if not route_data:
                    st.error("Could not calculate route. Please try different locations or travel mode.")
                else:
                    # Extract route information
                    route = route_data["routes"][0]
                    distance_km = route["summary"]["distance"] / 1000
                    duration_min = route["summary"]["duration"] / 60
                    
                    # Get AQI data for both locations
                    start_aqi_data = get_aqi_data(start_coords[0], start_coords[1])
                    end_aqi_data = get_aqi_data(end_coords[0], end_coords[1])
                    
                    if not start_aqi_data:
                        st.error(f"Could not retrieve air quality data for '{start_point}'.")
                    elif not end_aqi_data:
                        st.error(f"Could not retrieve air quality data for '{destination}'.")
                    else:
                        # Calculate carbon footprint
                        carbon_footprint = calculate_carbon_footprint(distance_km, selected_mode)
                        
                        # Display route information in a card
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.subheader("🛣️ Route Information")
                        
                        # Create columns for metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Distance", f"{distance_km:.2f} km")
                        with col2:
                            st.metric("Duration", f"{duration_min:.1f} min")
                        with col3:
                            if carbon_footprint is not None:
                                st.metric("Carbon Footprint", f"{carbon_footprint:.2f} kg CO₂")
                            else:
                                st.metric("Carbon Footprint", "Data unavailable")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Display AQI information in a card
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.subheader("🌬️ Air Quality Along Route")
                        
                        # Create columns for start and end AQI
                        col1, col2 = st.columns(2)
                        
                        # Extract AQI values
                        start_aqi = start_aqi_data["list"][0]["main"]["aqi"]
                        end_aqi = end_aqi_data["list"][0]["main"]["aqi"]
                        
                        # Map AQI value (1-5) to a more detailed scale (0-500)
                        start_aqi_value = start_aqi * 100 - 50
                        end_aqi_value = end_aqi * 100 - 50
                        
                        # Get AQI categories
                        start_category, start_color = get_aqi_category(start_aqi_value)
                        end_category, end_color = get_aqi_category(end_aqi_value)
                        
                        with col1:
                            st.markdown(f"<h3 style='text-align: center; color: {start_color};'>{start_category}</h3>", unsafe_allow_html=True)
                            st.markdown(f"<p style='text-align: center;'>Starting Point AQI: <span style='color: {start_color};'>{start_aqi}</span></p>", unsafe_allow_html=True)
                            
                            # Display pollutant levels
                            pollutants = start_aqi_data["list"][0]["components"]
                            for pollutant, value in pollutants.items():
                                st.markdown(f"- {pollutant.upper()}: {value:.1f} μg/m³")
                        
                        with col2:
                            st.markdown(f"<h3 style='text-align: center; color: {end_color};'>{end_category}</h3>", unsafe_allow_html=True)
                            st.markdown(f"<p style='text-align: center;'>Destination AQI: <span style='color: {end_color};'>{end_aqi}</span></p>", unsafe_allow_html=True)
                            
                            # Display pollutant levels
                            pollutants = end_aqi_data["list"][0]["components"]
                            for pollutant, value in pollutants.items():
                                st.markdown(f"- {pollutant.upper()}: {value:.1f} μg/m³")
                        
                        # Travel recommendations based on AQI and mode
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Recommendations Card
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.subheader("🌿 Eco-friendly Travel Recommendations")
                        
                        # Recommendations based on travel mode and AQI
                        if selected_mode == "Walking":
                            if start_aqi <= 2 and end_aqi <= 2:  # Good to Moderate
                                st.success("✅ The air quality is good for walking along this route!")
                                st.markdown("""
                                **Enjoy your walk with these tips:**
                                - Stay hydrated
                                - Use sunscreen
                                - Take breaks in shaded areas
                                - Consider a hat or sunglasses
                                """)
                            else:
                                st.warning("⚠️ Air quality along this route may not be ideal for walking.")
                                st.markdown("""
                                **Recommendations:**
                                - Consider wearing an N95 mask
                                - Try to walk during early morning or evening when pollution is lower
                                - Stay hydrated
                                - Consider alternative transportation if you have respiratory conditions
                                """)
                        
                        elif selected_mode == "Cycling":
                            if start_aqi <= 2 and end_aqi <= 2:
                                st.success("✅ Cycling is a great eco-friendly option for this route!")
                                st.markdown("""
                                **Tips for your ride:**
                                - Wear a helmet
                                - Use bike lanes when available
                                - Stay hydrated
                                - Consider a mask if you have respiratory sensitivities
                                """)
                            else:
                                st.warning("⚠️ Air quality along this route may affect your cycling experience.")
                                st.markdown("""
                                **Recommendations:**
                                - Wear an N95 mask while cycling
                                - Cycle during off-peak pollution hours
                                - Take breaks in green areas
                                - Consider a different mode of transport if you have respiratory conditions
                                """)
                        
                        else:  # Driving
                            st.info("ℹ️ While driving produces more emissions, here are ways to minimize your impact:")
                            st.markdown("""
                            **Eco-driving tips:**
                            - Maintain a steady speed
                            - Avoid rapid acceleration and braking
                            - Keep windows closed in high pollution areas
                            - Use recirculated air mode for your AC in high pollution areas
                            - Consider carpooling to reduce per-person emissions
                            """)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Please enter both starting point and destination.")
