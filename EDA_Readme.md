# EDA

1. **Data Overview** \
	•	Dataset Scale: The dataset includes over 3.9 million entries, covering various real estate metrics like trade prices, land area, and building year across Japan.  \
	•	Key Features:  \
	•	Trade Price  \
	•	Unit Price  \
	•	Area  \
	•	Proximity to nearest station  \
	•	Structure and building year  \
	•	Regional classifications (Prefecture, Municipality, and Region)  

2. **Real Estate Price Distribution**  \
	•	National Trends: \
	•	The median trade price is ¥14 million, with a broad range indicating significant variability across regions. \
	•	Trade prices range from ¥100 to a maximum of ¥61 billion, reflecting urban-rural disparities and differences in property types (residential vs. commercial). \
	•	Regional Insights: \
	•	Prefectures like Hokkaido and Okinawa show contrasting dynamics, with industrial and residential prices showing clear gaps. \
	•	Areas closer to urban centers and train stations command higher trade prices.

3. **Land and Building Characteristics** \
	•	Area Distribution: \
	•	Median property area is 185 m², with a large variation from small urban plots to expansive rural lands (max: 5000 m²). \
	•	Properties with larger areas are primarily located in rural or industrial regions. \
	•	Building Year: \
	•	Missing data for 52% of entries, but existing data suggests newer constructions (post-2000) have higher trade prices. \
	•	Older structures tend to depreciate unless located in prime regions like Tokyo or Osaka.

4. **Proximity to Nearest Station** \
	•	Properties with a shorter travel time to the nearest station (0-15 minutes) exhibit significantly higher trade prices. \
	•	Median travel time is 16 minutes. \
	•	Urban properties benefit from proximity to well-connected transport hubs. \

5. **Data Completeness and Gaps** \
	•	Significant missing data for critical fields: \
	•	Purpose: Missing in 69.7% of entries. \
	•	Building Year and Total Floor Area:  \Missing in over 50% of entries, limiting insights into structural trends. \
	•	Missing values for floor area ratio and coverage ratio are present but less significant.

6. **Visualization and Insights** \
	•	Choropleth Maps: \
	•	Prefectures were color-coded based on trade prices: \
	•	Green regions (e.g., Tokyo, Osaka) indicate high trade prices. \
	•	Red regions (e.g., rural prefectures) show significantly lower prices. \
	•	Histograms and Boxplots: \
	•	Skewed distributions in trade price and area highlight the influence of outliers (e.g., luxurious urban apartments and expansive rural properties). \
	•	Boxplots reveal a significant number of outliers, particularly in urban trade prices. 

7. **Key Factors Affecting Trade Prices** \
	•	Region and Proximity:  \
	•	Properties in urban centers and areas closer to transportation infrastructure fetch higher prices. \
	•	Type and Classification: \
	•	Commercial and residential properties dominate high-value transactions. \
	•	Structure and Age: \
	•	Modern, reinforced concrete structures have better valuations compared to older wooden constructions. 