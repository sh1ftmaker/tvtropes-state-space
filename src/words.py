"""Primitive narrative "words" — the base crafting ingredients for the game.

The crafting tab (craft.html) is an Infinite-Craft-style game: you combine two
ingredients and the result is whichever TVTrope sits nearest to the *sum* of the
two ingredient vectors in embedding space. The starting palette is genres (see
genres.py) plus the evocative story primitives below — heroes, deaths, swords,
curses, betrayals — so the very first combinations already reach into the trope
cloud.

Each entry is (name, kind, gloss). As with genres, the gloss gives
gemini-embedding-2 enough context to place the bare word meaningfully ("Mask"
alone is ambiguous; "Mask: a disguise hiding a character's true identity" is
not). `kind` only drives chip coloring in the UI.
"""
from __future__ import annotations

# kind -> display label + hex color (chip color for non-genre ingredients)
KINDS = {
    "character": ("Character", "#f472b6"),
    "concept":   ("Concept",   "#7cf2d6"),
    "object":    ("Object",    "#fbbf24"),
    "place":     ("Place",     "#38bdf8"),
    "event":     ("Event",     "#a78bfa"),
}

# (name, kind, gloss)
WORDS = [
    # ---------------- characters ----------------
    ("Hero", "character", "Hero: the courageous protagonist who rises to face danger and save others."),
    ("Villain", "character", "Villain: the evil antagonist who opposes the hero and threatens the world."),
    ("Mentor", "character", "Mentor: a wise older guide who trains and counsels the hero."),
    ("Sidekick", "character", "Sidekick: a loyal companion who supports the hero on the adventure."),
    ("Child", "character", "Child: a young innocent whose vulnerability raises the stakes."),
    ("Mother", "character", "Mother: a maternal figure of love, protection or grief."),
    ("Father", "character", "Father: a paternal figure of authority, legacy or estrangement."),
    ("King", "character", "King: a ruling monarch wielding power over a realm."),
    ("Queen", "character", "Queen: a sovereign woman of power, scheming or grace."),
    ("Knight", "character", "Knight: an armored warrior sworn to honor, duty and a code."),
    ("Wizard", "character", "Wizard: a master of arcane magic and ancient secrets."),
    ("Detective", "character", "Detective: a sharp investigator who unravels crimes by deduction."),
    ("Soldier", "character", "Soldier: a fighter bound to duty amid the horror of war."),
    ("Monster", "character", "Monster: a terrifying creature that menaces humanity."),
    ("Robot", "character", "Robot: an artificial mechanical being, servant or threat."),
    ("Ghost", "character", "Ghost: the restless spirit of the dead haunting the living."),
    ("God", "character", "God: a divine all-powerful being shaping fate and creation."),
    ("Demon", "character", "Demon: an infernal being of temptation and destruction."),
    ("Witch", "character", "Witch: a practitioner of dark or folk magic, feared and powerful."),
    ("Pirate", "character", "Pirate: a high-seas outlaw chasing plunder and freedom."),
    ("Spy", "character", "Spy: a secret agent of espionage, lies and double-crosses."),
    ("Outlaw", "character", "Outlaw: a fugitive living beyond the law on the frontier."),
    ("Rebel", "character", "Rebel: an insurgent rising up against a tyrannical regime."),
    ("Assassin", "character", "Assassin: a hired killer who strikes from the shadows."),

    # ---------------- concepts ----------------
    ("Love", "concept", "Love: passionate romantic devotion between two people."),
    ("Death", "concept", "Death: the end of life, grief and mortality."),
    ("Betrayal", "concept", "Betrayal: a trusted ally turning against the protagonist."),
    ("Revenge", "concept", "Revenge: a burning quest to avenge a past wrong."),
    ("Sacrifice", "concept", "Sacrifice: giving up one's life or happiness for a greater good."),
    ("Honor", "concept", "Honor: a code of integrity, duty and reputation worth dying for."),
    ("Destiny", "concept", "Destiny: a fated path the hero is bound to fulfill."),
    ("Power", "concept", "Power: the corrupting hunger for control and dominance."),
    ("Secret", "concept", "Secret: a hidden truth concealed at great cost."),
    ("Truth", "concept", "Truth: the revelation that exposes lies and changes everything."),
    ("Madness", "concept", "Madness: a mind unraveling into insanity and delusion."),
    ("Fear", "concept", "Fear: dread and terror that grips the heart."),
    ("Hope", "concept", "Hope: the stubborn faith that keeps the desperate going."),
    ("Memory", "concept", "Memory: the past remembered, lost or falsified."),
    ("Time", "concept", "Time: the passage, loops and paradoxes of time itself."),
    ("Justice", "concept", "Justice: the moral demand that wrongdoers answer for their crimes."),
    ("Freedom", "concept", "Freedom: the fight for liberty against oppression."),
    ("Greed", "concept", "Greed: an insatiable hunger for wealth and more."),
    ("Redemption", "concept", "Redemption: a fallen soul earning forgiveness through change."),
    ("Identity", "concept", "Identity: the question of who a character truly is."),
    ("Prophecy", "concept", "Prophecy: a foretold destiny that drives the plot."),
    ("Curse", "concept", "Curse: a supernatural affliction laid upon a person or place."),

    # ---------------- objects ----------------
    ("Sword", "object", "Sword: a bladed weapon of combat, honor and legend."),
    ("Gun", "object", "Gun: a firearm of violence, threat and sudden death."),
    ("Magic", "object", "Magic: supernatural power bending the laws of reality."),
    ("Machine", "object", "Machine: a device or technology that transforms the world."),
    ("Crown", "object", "Crown: a symbol of rulership, ambition and succession."),
    ("Treasure", "object", "Treasure: hidden riches that lure adventurers and thieves."),
    ("Potion", "object", "Potion: a brewed elixir of healing, poison or transformation."),
    ("Mask", "object", "Mask: a disguise hiding a character's true identity."),
    ("Mirror", "object", "Mirror: a reflection revealing the self, the double or another world."),
    ("Blood", "object", "Blood: violence, kinship and sacrifice made physical."),
    ("Letter", "object", "Letter: a written message carrying secrets, love or doom."),
    ("Map", "object", "Map: a guide to hidden places and the journey ahead."),

    # ---------------- places ----------------
    ("Castle", "place", "Castle: a fortified stronghold of nobles, sieges and intrigue."),
    ("City", "place", "City: a sprawling urban world of crowds, crime and ambition."),
    ("Forest", "place", "Forest: a wild, enchanted or threatening wilderness."),
    ("Ocean", "place", "Ocean: the vast sea of voyages, storms and the unknown."),
    ("Ruins", "place", "Ruins: the haunted remnants of a lost civilization."),
    ("Space", "place", "Space: the cosmic frontier of stars, ships and aliens."),
    ("Dungeon", "place", "Dungeon: a dark underground labyrinth of monsters and danger."),
    ("Wasteland", "place", "Wasteland: a desolate ruined land after catastrophe."),

    # ---------------- events ----------------
    ("War", "event", "War: large-scale armed conflict and its human cost."),
    ("Quest", "event", "Quest: a perilous journey toward a vital goal."),
    ("Heist", "event", "Heist: an elaborate, clever robbery pulled off by a crew."),
    ("Duel", "event", "Duel: a one-on-one fight of honor or vengeance."),
    ("Wedding", "event", "Wedding: a marriage ceremony of union, alliance or disaster."),
    ("Escape", "event", "Escape: a desperate flight from captivity or doom."),
    ("Apocalypse", "event", "Apocalypse: the catastrophic end of the world."),
    ("Invasion", "event", "Invasion: a hostile force overrunning a land or world."),
    ("Trial", "event", "Trial: a courtroom reckoning of guilt and judgment."),
    ("Rebirth", "event", "Rebirth: resurrection, renewal or transformation into something new."),
]


def word_text(name: str, gloss: str) -> str:
    """The string actually sent to the embedding model for a word."""
    return f"{name}. {gloss}"
