import re
import json
from typing import Dict, Optional, List, Tuple

class NaturalLanguageProcessor:
    def __init__(self):
        # Brand aliases and variations
        self.brand_aliases = {
            'Zara': ['zara'],
            'H&M': ['h&m', 'hm', 'h and m', 'hennes mauritz'],
            'Uniqlo': ['uniqlo', 'uniqulo'],
            'Gap': ['gap'],
            'Old Navy': ['old navy', 'oldnavy'],
            'Shein': ['shein', 'she in']
        }
        
    def estimate_body_measurements(self, height_cm: float, weight_kg: float) -> Dict[str, float]:
        """
        Estimate chest, waist, and hips based on height and weight
        Using realistic body proportion formulas
        """
        bmi = weight_kg / ((height_cm / 100) ** 2)
        
        # Estimation formulas (these are approximations)
        # For larger people, proportions change
        if bmi < 25: 
            chest = height_cm * 0.52 + weight_kg * 0.3
            waist = height_cm * 0.42 + weight_kg * 0.4
            hips = height_cm * 0.54 + weight_kg * 0.35
        elif bmi < 30:
            chest = height_cm * 0.54 + weight_kg * 0.35
            waist = height_cm * 0.45 + weight_kg * 0.5
            hips = height_cm * 0.56 + weight_kg * 0.4
        else:
            chest = height_cm * 0.56 + weight_kg * 0.4
            waist = height_cm * 0.48 + weight_kg * 0.6
            hips = height_cm * 0.58 + weight_kg * 0.45
        
        return {
            'chest': round(chest),
            'waist': round(waist),
            'hips': round(hips)
        }
        
    def extract_measurements(self, text: str) -> Dict[str, Optional[float]]:
        """Extract height, weight, and other measurements from natural language text."""
        text = text.lower().strip()
        measurements = {
            'height': None,
            'weight': None,
            'chest': None,
            'waist': None,
            'hips': None
        }
        
        # Extract height
        height = self._extract_height(text)
        if height:
            measurements['height'] = height
            
        # Extract weight
        weight = self._extract_weight(text)
        if weight:
            measurements['weight'] = weight
            
        # Extract chest, waist, hips if mentioned
        chest = self._extract_body_measurement(text, ['chest', 'bust'])
        if chest:
            measurements['chest'] = chest
            
        waist = self._extract_body_measurement(text, ['waist'])
        if waist:
            measurements['waist'] = waist
            
        hips = self._extract_body_measurement(text, ['hips', 'hip'])
        if hips:
            measurements['hips'] = hips
        
        # If we have height and weight but missing body measurements, estimate them
        if measurements['height'] and measurements['weight']:
            if not measurements['chest'] or not measurements['waist'] or not measurements['hips']:
                estimated = self.estimate_body_measurements(measurements['height'], measurements['weight'])
                if not measurements['chest']:
                    measurements['chest'] = estimated['chest']
                if not measurements['waist']:
                    measurements['waist'] = estimated['waist']
                if not measurements['hips']:
                    measurements['hips'] = estimated['hips']
        
        return measurements
    
    def _extract_height(self, text: str) -> Optional[float]:
        """Extract height from text and convert to cm."""
        # Pattern for feet and inches (e.g., "6 foot", "6'0", "6 feet", "6 ft")
        feet_patterns = [
            r"(\d+)\s*(?:foot|feet|ft)\s*(?:tall)?",  # "6 foot", "6 feet"
            r"(\d+)'(?:\s*(\d+))?",  # "6'0", "6'"
            r"(\d+)\s*(?:feet|ft)\s*(\d+)\s*(?:inches|in|\")?",  # "6 feet 0 inches"
        ]
        
        for pattern in feet_patterns:
            match = re.search(pattern, text)
            if match:
                feet = int(match.group(1))
                inches = 0
                if match.lastindex and match.lastindex > 1 and match.group(2):
                    inches = int(match.group(2))
                total_inches = feet * 12 + inches
                return round(total_inches * 2.54)  # Convert to cm
        
        # Pattern for feet and inches with explicit format
        feet_inches_patterns = [
            r"(\d+)'(\d+)",  # 6'0
            r"(\d+)\s*(?:feet|ft)\s*(\d+)\s*(?:inches|in|\")",  # 6 feet 0 inches
        ]
        
        for pattern in feet_inches_patterns:
            match = re.search(pattern, text)
            if match:
                feet = int(match.group(1))
                inches = int(match.group(2))
                total_inches = feet * 12 + inches
                return round(total_inches * 2.54)  # Convert to cm
        
        # Pattern for decimal feet (e.g., "5.5 feet")
        decimal_feet_pattern = r"(\d+\.?\d*)\s*(?:feet|ft)\b"
        match = re.search(decimal_feet_pattern, text)
        if match:
            feet = float(match.group(1))
            return round(feet * 30.48)  # Convert to cm
        
        # Pattern for cm (e.g., "180cm", "180 cm")
        cm_pattern = r"(\d+)\s*cm\b"
        match = re.search(cm_pattern, text)
        if match:
            return float(match.group(1))
        
        # Pattern for just numbers that might be height (between 140-220 cm range)
        height_patterns = [
            r"\b(1[4-9]\d|2[0-2]\d)\b(?!\s*(?:kg|lbs?|pounds))",  # 140-229, not followed by weight units
        ]
        
        for pattern in height_patterns:
            matches = re.findall(pattern, text)
            if matches:
                # Take the first reasonable height value
                height = int(matches[0])
                if 140 <= height <= 220:
                    return float(height)
        
        return None
    
    def _extract_weight(self, text: str) -> Optional[float]:
        """Extract weight from text and convert to kg."""
        # Pattern for pounds (e.g., "300 lbs", "300 pounds", "300 pound")
        pounds_patterns = [
            r"(\d+\.?\d*)\s*(?:lbs?|pounds?)\b",
        ]
        
        for pattern in pounds_patterns:
            match = re.search(pattern, text)
            if match:
                pounds = float(match.group(1))
                return round(pounds * 0.453592, 1)  # Convert to kg
        
        # Pattern for kg (e.g., "70kg", "70 kg", "70 kilos")
        kg_patterns = [
            r"(\d+\.?\d*)\s*(?:kg|kilos?|kilograms?)\b",
        ]
        
        for pattern in kg_patterns:
            match = re.search(pattern, text)
            if match:
                return float(match.group(1))
        
        # Look for weight-related context clues
        weight_context_patterns = [
            r"(?:weigh|weight)\s+(?:is\s+)?(\d+\.?\d*)\s*(?:lbs?|pounds?|kg|kilos?)?",
            r"(\d+\.?\d*)\s*(?:lbs?|pounds?|kg|kilos?)\s+(?:weight|heavy)",
        ]
        
        for pattern in weight_context_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                weight = float(match.group(1))
                # Try to determine if it's lbs or kg based on reasonable ranges
                if 30 <= weight <= 200:  # Could be kg
                    return weight
                elif 60 <= weight <= 500:  # Likely lbs
                    return round(weight * 0.453592, 1)
        
        return None
    
    def _extract_body_measurement(self, text: str, measurement_names: List[str]) -> Optional[float]:
        """Extract specific body measurements like chest, waist, hips."""
        for name in measurement_names:
            # Pattern for measurement with units
            patterns = [
                rf"{name}\s*:?\s*(\d+\.?\d*)\s*(?:cm|inches?|in)",
                rf"(\d+\.?\d*)\s*(?:cm|inches?|in)\s*{name}",
                rf"{name}\s*(?:is|measures?)?\s*(\d+\.?\d*)",
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    value = float(match.group(1))
                    # Convert inches to cm if needed
                    if 'inch' in text or 'in' in text:
                        value = value * 2.54
                    return round(value)
        
        return None
    
    def extract_brand(self, text: str) -> str:
        """Extract brand name from text."""
        text = text.lower()
        
        # Look for patterns like "at zara", "from h&m", "for uniqlo"
        brand_patterns = [
            r"(?:at|from|for|in|shop|shopping|buy|buying|want|wanna)\s+([a-z&\s]+)",
            r"([a-z&\s]+)\s+(?:store|shop|brand|clothing)",
        ]
        
        for pattern in brand_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                brand_candidate = match.strip()
                normalized_brand = self._normalize_brand(brand_candidate)
                if normalized_brand:
                    return normalized_brand
        
        # Check for direct brand mentions
        for brand, aliases in self.brand_aliases.items():
            for alias in aliases:
                if alias in text:
                    return brand
        
        # Default brand
        return "Zara"
    
    def _normalize_brand(self, brand_text: str) -> Optional[str]:
        """Normalize brand text to standard brand names."""
        brand_text = brand_text.lower().strip()
        
        for brand, aliases in self.brand_aliases.items():
            if brand_text in aliases:
                return brand
        
        return None
    
    def parse_input(self, user_input: str) -> Dict:
        """Main function to parse natural language input."""
        measurements = self.extract_measurements(user_input)
        brand = self.extract_brand(user_input)
        
        return {
            'height': measurements['height'],
            'weight': measurements['weight'],
            'chest': measurements['chest'],
            'waist': measurements['waist'],
            'hips': measurements['hips'],
            'brand': brand,
            'original_input': user_input
        }
    
    def validate_input(self, parsed_data: Dict) -> Tuple[bool, List[str]]:
        """Validate that required measurements are present."""
        errors = []
        
        if not parsed_data['height']:
            errors.append("I couldn't find your height. Please include it (e.g., '6 foot' or '183cm')")
        
        if not parsed_data['weight']:
            errors.append("I couldn't find your weight. Please include it (e.g., '300 lbs' or '136 kg')")
        
        return len(errors) == 0, errors

# Test the processor
if __name__ == "__main__":
    processor = NaturalLanguageProcessor()
    
    test_inputs = [
        "Im 6 foot tall and im 300 pounds, i wanna shop at Uniqlo, whats my size?",
        "I'm 5'8 and weigh 150 lbs, looking for clothes at Zara",
        "170cm, 65kg, shopping at H&M",
        "I am 5 feet 6 inches tall, 140 pounds, need size for Uniqlo",
        "Height 168cm weight 58kg chest 86cm waist 68cm hips 92cm for Gap",
        "I'm about 5'9, around 160 lbs, want to buy from Shein",
        "Looking for old navy clothes, I'm 175cm and 70kg"
    ]
    
    print("Testing Updated Natural Language Processor:")
    print("=" * 60)