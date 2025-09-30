"""MOFMaster Agent - Primary MOF generation agent.

This agent provides three main generation modes:
1. Basic generation without chemical constraints
2. Metal-specific generation with specific SBU
3. Composition-specific generation with target composition
"""

import os
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from openai import OpenAI


class MOFGenerationMode(Enum):
    """Enumeration of MOF generation modes."""
    BASIC = "basic"  # Generate without chemical constraints
    METAL_SPECIFIC = "metal_specific"  # Generate with specific metal SBU
    COMPOSITION_SPECIFIC = "composition_specific"  # Generate with target composition


@dataclass
class MOFStructure:
    """Represents a MOF structure with its properties."""
    cif_content: str
    formula: str
    metal_sbu: Optional[str] = None
    composition: Optional[Dict[str, float]] = None
    properties: Optional[Dict[str, Any]] = None
    generation_metadata: Optional[Dict[str, Any]] = None


@dataclass
class MOFGenerationRequest:
    """Request for MOF generation."""
    mode: MOFGenerationMode
    count: int = 1
    metal: Optional[str] = None
    composition: Optional[Dict[str, float]] = None
    constraints: Optional[Dict[str, Any]] = None
    target_properties: Optional[Dict[str, Any]] = None


@dataclass
class MOFGenerationResponse:
    """Response containing generated MOF structures."""
    structures: List[MOFStructure]
    generation_time: float
    success_count: int
    mode_used: MOFGenerationMode
    metadata: Dict[str, Any]


class MOFMaster:
    """MOFMaster Agent - Primary MOF generation agent.
    
    Provides three generation modes:
    1. Basic: Generate MOF structures without specific chemical constraints
    2. Metal-specific: Generate MOFs with specific metal secondary building units (SBUs)
    3. Composition-specific: Generate MOFs with target chemical composition
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        """Initialize MOFMaster with OpenAI configuration."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.generation_templates = self._load_generation_templates()
    
    def _load_generation_templates(self) -> Dict[str, str]:
        """Load generation templates for different modes."""
        return {
            "basic": """
Generate {count} novel Metal-Organic Framework (MOF) structures.

Requirements:
- Each structure should be chemically reasonable and stable
- Include diverse metal centers and organic linkers
- Provide CIF format for each structure
- Include chemical formula and basic properties

Format the response as:
Structure 1:
CIF: [CIF content]
Formula: [Chemical formula]
Properties: [Basic properties]

Structure 2:
...
""",
            "metal_specific": """
Generate {count} Metal-Organic Framework (MOF) structures containing {metal} as the primary metal in the secondary building unit (SBU).

Requirements:
- {metal} must be the primary metal center in the SBU
- Include diverse organic linkers that coordinate to {metal}
- Ensure chemical stability and reasonable coordination geometry
- Provide CIF format for each structure
- Include chemical formula and metal coordination details

Format the response as:
Structure 1:
CIF: [CIF content]
Formula: [Chemical formula]
Metal SBU: {metal} coordination details
Properties: [Basic properties]

Structure 2:
...
""",
            "composition_specific": """
Generate {count} Metal-Organic Framework (MOF) structures with the target composition: {composition}.

Requirements:
- Achieve the target elemental composition as closely as possible
- Maintain chemical stability and reasonable coordination
- Include diverse structural arrangements
- Provide CIF format for each structure
- Include actual vs target composition analysis

Format the response as:
Structure 1:
CIF: [CIF content]
Formula: [Chemical formula]
Target Composition: {composition}
Actual Composition: [Actual composition]
Properties: [Basic properties]

Structure 2:
...
"""
        }
    
    def generate_structures(self, request: MOFGenerationRequest) -> MOFGenerationResponse:
        """Generate MOF structures based on the request."""
        start_time = time.time()
        
        # Generate structures based on mode
        structures = self._generate_structures(request)
        
        generation_time = time.time() - start_time
        
        return MOFGenerationResponse(
            structures=structures,
            generation_time=generation_time,
            success_count=len(structures),
            mode_used=request.mode,
            metadata={
                "total_requested": request.count,
                "model_used": self.model,
                "generation_mode": request.mode.value
            }
        )
    
    def _generate_structures(self, request: MOFGenerationRequest) -> List[MOFStructure]:
        """Generate structures based on the request mode."""
        structures = []
        
        for i in range(request.count):
            if request.mode == MOFGenerationMode.BASIC:
                structure = self._generate_basic_structure(i + 1, request.constraints)
            elif request.mode == MOFGenerationMode.METAL_SPECIFIC:
                structure = self._generate_metal_specific_structure(
                    i + 1, request.metal, request.constraints
                )
            elif request.mode == MOFGenerationMode.COMPOSITION_SPECIFIC:
                structure = self._generate_composition_specific_structure(
                    i + 1, request.composition, request.constraints
                )
            else:
                raise ValueError(f"Unknown generation mode: {request.mode}")
            
            structures.append(structure)
        
        return structures
    
    def _generate_basic_structure(self, index: int, constraints: Optional[Dict[str, Any]]) -> MOFStructure:
        """Generate a basic MOF structure without specific constraints."""
        prompt = self.generation_templates["basic"].format(count=1)
        
        # Add constraints if provided
        if constraints:
            constraint_text = "\n".join([f"- {k}: {v}" for k, v in constraints.items()])
            prompt += f"\nAdditional constraints:\n{constraint_text}"
        
        response = self._generate_with_openai(prompt)
        
        # Parse response to extract structure information
        return self._parse_structure_response(response, index, "basic")
    
    def _generate_metal_specific_structure(
        self, 
        index: int, 
        metal: str, 
        constraints: Optional[Dict[str, Any]]
    ) -> MOFStructure:
        """Generate a MOF structure with specific metal SBU."""
        if not metal:
            raise ValueError("Metal must be specified for metal-specific generation")
        
        prompt = self.generation_templates["metal_specific"].format(
            count=1, metal=metal
        )
        
        # Add constraints if provided
        if constraints:
            constraint_text = "\n".join([f"- {k}: {v}" for k, v in constraints.items()])
            prompt += f"\nAdditional constraints:\n{constraint_text}"
        
        response = self._generate_with_openai(prompt)
        
        # Parse response to extract structure information
        structure = self._parse_structure_response(response, index, "metal_specific")
        structure.metal_sbu = metal
        return structure
    
    def _generate_composition_specific_structure(
        self, 
        index: int, 
        composition: Dict[str, float], 
        constraints: Optional[Dict[str, Any]]
    ) -> MOFStructure:
        """Generate a MOF structure with target composition."""
        if not composition:
            raise ValueError("Composition must be specified for composition-specific generation")
        
        prompt = self.generation_templates["composition_specific"].format(
            count=1, composition=composition
        )
        
        # Add constraints if provided
        if constraints:
            constraint_text = "\n".join([f"- {k}: {v}" for k, v in constraints.items()])
            prompt += f"\nAdditional constraints:\n{constraint_text}"
        
        response = self._generate_with_openai(prompt)
        
        # Parse response to extract structure information
        structure = self._parse_structure_response(response, index, "composition_specific")
        structure.composition = composition
        return structure
    
    def _generate_with_openai(self, prompt: str) -> str:
        """Generate text using OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in Metal-Organic Framework (MOF) chemistry and crystallography. Generate accurate and chemically reasonable MOF structures."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4000,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating with OpenAI: {e}")
            return "Error generating structure: " + str(e)
    
    def _parse_structure_response(
        self, 
        response: str, 
        index: int, 
        mode: str
    ) -> MOFStructure:
        """Parse LLM response to extract MOF structure information."""
        # This is a simplified parser - in practice, you'd want more robust parsing
        lines = response.split('\n')
        
        # Extract CIF content (simplified)
        cif_content = ""
        formula = ""
        properties = {}
        
        in_cif = False
        for line in lines:
            if "CIF:" in line:
                in_cif = True
                cif_content += line.replace("CIF:", "").strip() + "\n"
            elif "Formula:" in line:
                formula = line.replace("Formula:", "").strip()
                in_cif = False
            elif "Properties:" in line:
                properties_text = line.replace("Properties:", "").strip()
                properties = {"description": properties_text}
                in_cif = False
            elif in_cif and line.strip():
                cif_content += line + "\n"
        
        # Generate placeholder CIF if not found
        if not cif_content:
            cif_content = f"""# Generated MOF Structure {index}
# Mode: {mode}
data_{index}
_cell_length_a   10.0
_cell_length_b   10.0
_cell_length_c   10.0
_cell_angle_alpha   90.0
_cell_angle_beta    90.0
_cell_angle_gamma   90.0
_space_group_name_H-M    'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Zn1 Zn 0.0 0.0 0.0
C1 C 0.25 0.25 0.25
O1 O 0.5 0.5 0.5
"""
        
        # Generate placeholder formula if not found
        if not formula:
            formula = "ZnC4H4O4"  # Placeholder formula
        
        return MOFStructure(
            cif_content=cif_content,
            formula=formula,
            properties=properties,
            generation_metadata={
                "mode": mode,
                "index": index,
                "generated_by": "MOFMaster"
            }
        )
    
    # Convenience methods for different generation modes
    def generate_basic_structures(self, count: int = 1, **constraints) -> MOFGenerationResponse:
        """Generate basic MOF structures without specific constraints."""
        request = MOFGenerationRequest(
            mode=MOFGenerationMode.BASIC,
            count=count,
            constraints=constraints
        )
        return self.generate_structures(request)
    
    def generate_metal_specific_structures(
        self, 
        metal: str, 
        count: int = 1, 
        **constraints
    ) -> MOFGenerationResponse:
        """Generate MOF structures with specific metal SBU."""
        request = MOFGenerationRequest(
            mode=MOFGenerationMode.METAL_SPECIFIC,
            count=count,
            metal=metal,
            constraints=constraints
        )
        return self.generate_structures(request)
    
    def generate_composition_specific_structures(
        self, 
        composition: Dict[str, float], 
        count: int = 1, 
        **constraints
    ) -> MOFGenerationResponse:
        """Generate MOF structures with target composition."""
        request = MOFGenerationRequest(
            mode=MOFGenerationMode.COMPOSITION_SPECIFIC,
            count=count,
            composition=composition,
            constraints=constraints
        )
        return self.generate_structures(request)


