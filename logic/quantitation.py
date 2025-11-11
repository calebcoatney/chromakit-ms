"""
Quantitation logic for Polyarc + Internal Standard method.

This module provides quantitation calculations for GC-MS data using the Polyarc
reactor approach, where all carbon-containing species are converted to CH4 and
measured on a per-carbon basis, enabling universal response factor quantitation.
"""

import re
from typing import Optional, Dict, List, Tuple


class QuantitationCalculator:
    """Calculator for Polyarc + Internal Standard quantitation method."""
    
    def __init__(self):
        self.internal_standard = None
        self.response_factor = None
        self.sample_info = {}
        
    def extract_carbon_count(self, formula: str) -> Optional[int]:
        """
        Extract the number of carbon atoms from a molecular formula.
        
        Args:
            formula: Molecular formula string (e.g., 'C9H20', 'CH3OH', 'CO2')
            
        Returns:
            Number of carbon atoms, or None if formula cannot be parsed
            
        Examples:
            'C9H20' -> 9
            'C6H6' -> 6
            'CO2' -> 1
            'CH3OH' -> 1
        """
        if not formula or not isinstance(formula, str):
            return None
            
        # Remove whitespace
        formula = formula.strip()
        
        # Pattern to match C followed by optional number
        # Matches: C, C9, C10, etc.
        pattern = r'C(\d*)'
        match = re.search(pattern, formula)
        
        if not match:
            return None
            
        carbon_str = match.group(1)
        
        # If no number after C, it means 1 carbon
        if carbon_str == '':
            return 1
        else:
            try:
                return int(carbon_str)
            except ValueError:
                return None
    
    def calculate_mol_C_internal_standard(
        self,
        volume_uL: float,
        density_g_mL: float,
        molecular_weight: float,
        formula: str
    ) -> Optional[float]:
        """
        Calculate moles of carbon in the internal standard.
        
        Args:
            volume_uL: Volume of IS added in microliters
            density_g_mL: Density of IS in g/mL
            molecular_weight: Molecular weight of IS in g/mol
            formula: Molecular formula of IS
            
        Returns:
            Moles of carbon (mol C), or None if calculation fails
        """
        try:
            # Extract number of carbons
            num_carbons = self.extract_carbon_count(formula)
            if num_carbons is None or num_carbons == 0:
                return None
            
            # Convert volume to mL
            volume_mL = volume_uL * 1e-3
            
            # Calculate mass of IS in grams
            mass_g = volume_mL * density_g_mL
            
            # Calculate moles of IS
            mol_IS = mass_g / molecular_weight
            
            # Calculate moles of carbon
            mol_C = mol_IS * num_carbons
            
            return mol_C
            
        except (ValueError, ZeroDivisionError, TypeError):
            return None
    
    def calculate_response_factor(
        self,
        is_area: float,
        mol_C_IS: float
    ) -> Optional[float]:
        """
        Calculate the universal response factor from internal standard.
        
        Args:
            is_area: FID peak area of internal standard
            mol_C_IS: Moles of carbon in internal standard
            
        Returns:
            Response factor (Area / mol C), or None if calculation fails
        """
        try:
            if mol_C_IS <= 0:
                return None
            return is_area / mol_C_IS
        except (ValueError, ZeroDivisionError, TypeError):
            return None
    
    def calculate_sample_mass(
        self,
        volume_uL: float,
        density_g_mL: Optional[float]
    ) -> Optional[float]:
        """
        Calculate sample mass from volume and density.
        
        Args:
            volume_uL: Sample volume in microliters
            density_g_mL: Sample density in g/mL (optional)
            
        Returns:
            Sample mass in mg, or None if density not provided
        """
        if density_g_mL is None:
            return None
            
        try:
            # Convert to mL
            volume_mL = volume_uL * 1e-3
            
            # Calculate mass in mg
            mass_mg = volume_mL * density_g_mL * 1000
            
            return mass_mg
            
        except (ValueError, TypeError):
            return None
    
    def quantitate_peak(
        self,
        peak_area: float,
        response_factor: float,
        formula: str,
        molecular_weight: float
    ) -> Optional[Dict[str, float]]:
        """
        Quantitate a single peak using the response factor.
        
        Args:
            peak_area: FID peak area
            response_factor: Universal response factor (Area / mol C)
            formula: Molecular formula of compound
            molecular_weight: Molecular weight in g/mol
            
        Returns:
            Dictionary with quantitation results, or None if calculation fails
            Keys: 'mol_C', 'num_carbons', 'mol', 'mass_mg'
        """
        try:
            # Extract number of carbons
            num_carbons = self.extract_carbon_count(formula)
            if num_carbons is None or num_carbons == 0:
                return None
            
            # Calculate mol C
            mol_C = peak_area / response_factor
            
            # Calculate moles of compound
            mol = mol_C / num_carbons
            
            # Calculate mass in mg
            mass_mg = mol * molecular_weight * 1000
            
            return {
                'mol_C': mol_C,
                'num_carbons': num_carbons,
                'mol': mol,
                'mass_mg': mass_mg
            }
            
        except (ValueError, ZeroDivisionError, TypeError):
            return None
    
    def calculate_composition(
        self,
        peaks_data: List[Dict]
    ) -> List[Dict]:
        """
        Calculate mol_C%, mol%, and wt% for all quantitated peaks.
        
        Args:
            peaks_data: List of dictionaries with 'mol_C', 'mol' and 'mass_mg' keys
            
        Returns:
            Updated list with 'mol_C_percent', 'mol_percent' and 'wt_percent' added
        """
        try:
            # Calculate totals
            total_mol_C = sum(p.get('mol_C', 0) for p in peaks_data if p.get('mol_C') is not None)
            total_mol = sum(p.get('mol', 0) for p in peaks_data if p.get('mol') is not None)
            total_mass = sum(p.get('mass_mg', 0) for p in peaks_data if p.get('mass_mg') is not None)
            
            if total_mol_C <= 0 or total_mol <= 0 or total_mass <= 0:
                return peaks_data
            
            # Calculate percentages
            for peak in peaks_data:
                if peak.get('mol_C') is not None:
                    peak['mol_C_percent'] = (peak['mol_C'] / total_mol_C) * 100
                else:
                    peak['mol_C_percent'] = None
                    
                if peak.get('mol') is not None:
                    peak['mol_percent'] = (peak['mol'] / total_mol) * 100
                else:
                    peak['mol_percent'] = None
                    
                if peak.get('mass_mg') is not None:
                    peak['wt_percent'] = (peak['mass_mg'] / total_mass) * 100
                else:
                    peak['wt_percent'] = None
            
            return peaks_data
            
        except (ValueError, ZeroDivisionError, TypeError):
            return peaks_data
    
    def calculate_carbon_balance(
        self,
        total_mass_mg: float,
        sample_mass_mg: Optional[float]
    ) -> Optional[float]:
        """
        Calculate carbon balance (recovery percentage).
        
        Args:
            total_mass_mg: Total quantitated mass in mg
            sample_mass_mg: Actual sample mass in mg (optional)
            
        Returns:
            Carbon balance as percentage, or None if sample mass not provided
        """
        if sample_mass_mg is None or sample_mass_mg <= 0:
            return None
            
        try:
            return (total_mass_mg / sample_mass_mg) * 100
        except (ValueError, ZeroDivisionError, TypeError):
            return None
    
    def validate_inputs(
        self,
        volume_uL: float,
        density_g_mL: float,
        molecular_weight: float
    ) -> Tuple[bool, str]:
        """
        Validate input parameters.
        
        Args:
            volume_uL: Volume in microliters
            density_g_mL: Density in g/mL
            molecular_weight: Molecular weight in g/mol
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if volume_uL <= 0:
            return False, "Volume must be greater than 0"
        
        if volume_uL > 1000:
            return False, "Volume seems unreasonably large (>1000 µL)"
        
        if density_g_mL <= 0:
            return False, "Density must be greater than 0"
        
        if density_g_mL < 0.5 or density_g_mL > 2.0:
            return False, "Density should be between 0.5 and 2.0 g/mL"
        
        if molecular_weight <= 0:
            return False, "Molecular weight must be greater than 0"
        
        if molecular_weight < 12:  # Less than carbon atom
            return False, "Molecular weight seems too low"
        
        return True, ""
