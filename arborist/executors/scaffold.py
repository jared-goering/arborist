"""Script scaffold management — mark sections as FROZEN vs MODIFIABLE."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class SectionType(str, Enum):
    """Whether a section can be modified by the LLM."""

    FROZEN = "FROZEN"
    MODIFIABLE = "MODIFIABLE"


@dataclass
class ScriptSection:
    """A labeled region of a script."""

    name: str
    section_type: SectionType
    content: str
    start_line: int = 0
    end_line: int = 0


@dataclass
class ScriptScaffold:
    """A script broken into labeled sections for controlled LLM modification.

    Sections are delimited by marker comments in the source:
        # --- SECTION: <name> [FROZEN|MODIFIABLE] ---
        ...code...
        # --- END SECTION ---

    Code outside any section markers is treated as FROZEN by default.
    """

    sections: list[ScriptSection] = field(default_factory=list)
    preamble: str = ""  # Code before the first section marker

    # Regex for section markers
    _SECTION_START = re.compile(
        r"^#\s*---\s*SECTION:\s*(.+?)\s*\[(FROZEN|MODIFIABLE)\]\s*---\s*$"
    )
    _SECTION_END = re.compile(r"^#\s*---\s*END SECTION\s*---\s*$")

    @classmethod
    def from_script(cls, script: str) -> ScriptScaffold:
        """Parse a script with section markers into a ScriptScaffold."""
        scaffold = cls()
        lines = script.split("\n")

        current_section: ScriptSection | None = None
        current_lines: list[str] = []
        preamble_lines: list[str] = []
        seen_any_section = False

        for i, line in enumerate(lines, 1):
            start_match = cls._SECTION_START.match(line.strip())
            end_match = cls._SECTION_END.match(line.strip())

            if start_match:
                # Close any open section
                if current_section:
                    current_section.content = "\n".join(current_lines)
                    current_section.end_line = i - 1
                    scaffold.sections.append(current_section)
                    current_lines = []
                elif not seen_any_section and preamble_lines:
                    scaffold.preamble = "\n".join(preamble_lines)
                    preamble_lines = []
                seen_any_section = True

                name = start_match.group(1).strip()
                stype = SectionType(start_match.group(2))
                current_section = ScriptSection(
                    name=name,
                    section_type=stype,
                    content="",
                    start_line=i + 1,
                )
            elif end_match and current_section:
                current_section.content = "\n".join(current_lines)
                current_section.end_line = i - 1
                scaffold.sections.append(current_section)
                current_section = None
                current_lines = []
            elif current_section:
                current_lines.append(line)
            else:
                preamble_lines.append(line)

        # Handle unclosed section or trailing preamble
        if current_section:
            current_section.content = "\n".join(current_lines)
            current_section.end_line = len(lines)
            scaffold.sections.append(current_section)
        elif preamble_lines and not scaffold.preamble:
            scaffold.preamble = "\n".join(preamble_lines)

        return scaffold

    def get_modifiable_sections(self) -> list[ScriptSection]:
        """Return only sections the LLM is allowed to change."""
        return [s for s in self.sections if s.section_type == SectionType.MODIFIABLE]

    def get_frozen_sections(self) -> list[ScriptSection]:
        """Return sections the LLM must not change."""
        return [s for s in self.sections if s.section_type == SectionType.FROZEN]

    def build_prompt_context(self) -> str:
        """Build a representation of the script for the LLM prompt.

        Shows all sections with clear labels about what can/cannot be changed.
        """
        parts = []
        if self.preamble.strip():
            parts.append(
                "# ========== PREAMBLE (FROZEN — do not modify) ==========\n"
                f"{self.preamble}\n"
            )

        for section in self.sections:
            label = section.section_type.value
            if section.section_type == SectionType.FROZEN:
                parts.append(
                    f"# ========== {section.name} ({label} — do not modify) ==========\n"
                    f"{section.content}\n"
                )
            else:
                parts.append(
                    f"# ========== {section.name} ({label} — you may modify this) ==========\n"
                    f"{section.content}\n"
                )

        return "\n".join(parts)

    def reassemble(self, modified_sections: dict[str, str]) -> str:
        """Reassemble the full script, replacing only modifiable sections.

        Args:
            modified_sections: Mapping of section name -> new content.
                Only MODIFIABLE sections are replaced; FROZEN sections
                are always preserved unchanged.

        Returns:
            The complete reassembled script.
        """
        parts = []
        if self.preamble.strip():
            parts.append(self.preamble)

        for section in self.sections:
            marker_start = f"# --- SECTION: {section.name} [{section.section_type.value}] ---"
            marker_end = "# --- END SECTION ---"
            parts.append(marker_start)

            if (
                section.section_type == SectionType.MODIFIABLE
                and section.name in modified_sections
            ):
                parts.append(modified_sections[section.name])
            else:
                parts.append(section.content)

            parts.append(marker_end)

        return "\n".join(parts)

    def verify_frozen_preserved(self, new_script: str) -> list[str]:
        """Check that all frozen sections in new_script match the originals.

        Returns a list of violation descriptions (empty = all good).
        """
        new_scaffold = ScriptScaffold.from_script(new_script)
        violations = []

        frozen_by_name = {s.name: s for s in self.get_frozen_sections()}
        new_by_name = {s.name: s for s in new_scaffold.sections}

        for name, original in frozen_by_name.items():
            if name not in new_by_name:
                violations.append(f"FROZEN section '{name}' is missing from output")
            elif new_by_name[name].content.strip() != original.content.strip():
                violations.append(f"FROZEN section '{name}' was modified")

        return violations
