"""Curated narrative genre / setting taxonomy for the TVTropes state space.

Drawn from Wikipedia's "List of writing genres"
(https://en.wikipedia.org/wiki/List_of_writing_genres), filtered to the
*narrative* / storytelling genres that actually apply to tropes (the page's
nonfiction branches -- theology, academic writing, obituaries, etc. -- are
dropped because they don't describe story devices).

Each entry is (name, supergenre, gloss). The gloss is a short descriptive
sentence: embedding the bare label ("Cyberpunk") is ambiguous, but embedding
"Cyberpunk: a science-fiction subgenre of high-tech, low-life near futures..."
gives gemini-embedding-2 enough context to place the genre meaningfully in the
same space as the tropes.

`super` groups each genre under one of a handful of supergenres. Those buckets
are aggregated into per-trope affinity scores and exposed as remappable axes in
the 3D view (set X=Fantasy, Y=Horror, Z=ScienceFiction and watch tropes
regroup).
"""
from __future__ import annotations

# supergenre -> display label + hex color (used for legends / coloring)
SUPERGENRES = {
    "Fantasy":         ("Fantasy",          "#8b5cf6"),
    "ScienceFiction":  ("Science Fiction",  "#22d3ee"),
    "Horror":          ("Horror",           "#ef4444"),
    "Romance":         ("Romance",          "#ec4899"),
    "CrimeMystery":    ("Crime & Mystery",  "#f59e0b"),
    "Thriller":        ("Thriller",         "#fb923c"),
    "ActionAdventure": ("Action & Adventure", "#84cc16"),
    "Comedy":          ("Comedy",           "#fde047"),
    "Drama":           ("Drama & Realist",  "#60a5fa"),
    "Historical":      ("Historical",       "#a8a29e"),
    "Western":         ("Western",          "#d97706"),
    "Superhero":       ("Superhero",        "#3b82f6"),
    "WarPolitical":    ("War & Political",   "#10b981"),
    "Myth":            ("Myth & Folklore",  "#c084fc"),
}

# (name, supergenre, gloss)
GENRES = [
    # ---------------- Fantasy ----------------
    ("High Fantasy", "Fantasy", "High fantasy: epic stories set in a fully invented secondary world with its own maps, races and magic systems, where heroes face a world-threatening evil."),
    ("Low Fantasy", "Fantasy", "Low fantasy: magic intrudes sparingly into an otherwise grounded, mundane world."),
    ("Sword and Sorcery", "Fantasy", "Sword and sorcery: pulpy adventures of roguish warriors and sorcerers facing personal stakes, monsters and dark magic."),
    ("Heroic Fantasy", "Fantasy", "Heroic fantasy: muscular barbarian-hero adventures of combat, treasure and sorcery."),
    ("Dark Fantasy", "Fantasy", "Dark fantasy: fantasy steeped in horror, dread and morally grim atmosphere."),
    ("Grimdark", "Fantasy", "Grimdark: cynical, violent fantasy where the world is brutal and no one is purely good."),
    ("Urban Fantasy", "Fantasy", "Urban fantasy: magic, monsters and the supernatural hidden inside a modern real-world city."),
    ("Contemporary Fantasy", "Fantasy", "Contemporary fantasy: magic and mythical beings existing in the present-day ordinary world."),
    ("Fairy Tale", "Fantasy", "Fairy tale: short wonder-story of enchantments, talking animals, witches, princes and moral lessons."),
    ("Magical Realism", "Fantasy", "Magic realism: realistic literary fiction in which magical events are treated as ordinary and unremarkable."),
    ("Fantasy of Manners", "Fantasy", "Fantasy of manners: intrigue, wit and social maneuvering among the aristocracy of a magical society."),
    ("Gaslamp Fantasy", "Fantasy", "Gaslamp fantasy: Victorian or Edwardian setting laced with magic and the occult."),
    ("Sword and Planet", "Fantasy", "Sword and planet: swashbuckling heroes wielding swords on exotic alien worlds."),
    ("Isekai Portal Fantasy", "Fantasy", "Isekai / portal fantasy: an ordinary person transported into a fantastical magical world or game-like realm."),
    ("Mythic Fantasy", "Fantasy", "Mythic fiction: stories rooted in legend, folklore and gods reworked into fantasy."),

    # ---------------- Science Fiction ----------------
    ("Hard Science Fiction", "ScienceFiction", "Hard science fiction: rigorous, physics-respecting speculation about technology and space."),
    ("Soft Science Fiction", "ScienceFiction", "Soft science fiction: SF focused on society, psychology and character rather than hard tech."),
    ("Space Opera", "ScienceFiction", "Space opera: grand galaxy-spanning adventure of starships, empires, aliens and epic battles."),
    ("Cyberpunk", "ScienceFiction", "Cyberpunk: high-tech low-life near future of hackers, megacorporations, cybernetics and neon dystopia."),
    ("Steampunk", "ScienceFiction", "Steampunk: retro-futurism powered by Victorian steam machinery and brass clockwork."),
    ("Dieselpunk", "ScienceFiction", "Dieselpunk: gritty interwar diesel-and-rivets retro-future aesthetic."),
    ("Biopunk", "ScienceFiction", "Biopunk: near-future fiction of genetic engineering, biotech and synthetic life."),
    ("Solarpunk", "ScienceFiction", "Solarpunk: optimistic future of renewable energy, ecology and sustainable community."),
    ("Post-Apocalyptic", "ScienceFiction", "Post-apocalyptic fiction: survival amid the ruins after civilization has collapsed."),
    ("Apocalyptic", "ScienceFiction", "Apocalyptic fiction: the catastrophic end of the world unfolding."),
    ("Dystopian", "ScienceFiction", "Dystopian fiction: oppressive future society of surveillance, control and lost freedom."),
    ("Utopian", "ScienceFiction", "Utopian fiction: an idealized, perfected future society."),
    ("Military Science Fiction", "ScienceFiction", "Military science fiction: future soldiers, fleets and interstellar warfare from a combatant's view."),
    ("Time Travel", "ScienceFiction", "Time travel fiction: journeys through time, paradoxes and altered histories."),
    ("Mecha", "ScienceFiction", "Mecha: giant piloted humanoid war machines and robots."),
    ("Alien Invasion", "ScienceFiction", "Alien invasion: extraterrestrials attacking or infiltrating Earth."),
    ("Climate Fiction", "ScienceFiction", "Climate fiction: stories of climate change and ecological catastrophe."),

    # ---------------- Horror ----------------
    ("Supernatural Horror", "Horror", "Supernatural horror: terror driven by ghosts, demons and forces beyond the natural world."),
    ("Psychological Horror", "Horror", "Psychological horror: dread built from paranoia, madness and the unraveling mind."),
    ("Body Horror", "Horror", "Body horror: revulsion at the mutilation, mutation or grotesque transformation of the body."),
    ("Cosmic Horror", "Horror", "Lovecraftian cosmic horror: insignificant humanity facing vast, unknowable, sanity-destroying entities."),
    ("Gothic Horror", "Horror", "Gothic fiction: brooding castles, decay, curses and romantic terror."),
    ("Slasher Splatter", "Horror", "Splatter / slasher horror: graphic gore and a killer stalking victims."),
    ("Ghost Story", "Horror", "Ghost story: hauntings by the restless dead."),
    ("Vampire Fiction", "Horror", "Vampire fiction: blood-drinking undead seducers and predators."),
    ("Werewolf Fiction", "Horror", "Werewolf fiction: humans cursed to transform into savage beasts."),
    ("Zombie Horror", "Horror", "Zombie fiction: the reanimated or infected dead overrunning the living."),
    ("Monster Horror", "Horror", "Monster literature: a terrifying creature menacing humanity."),
    ("Comedy Horror", "Horror", "Comedy horror: scares blended with laughs and the absurd."),

    # ---------------- Romance ----------------
    ("Contemporary Romance", "Romance", "Contemporary romance: present-day love stories of courtship and emotional connection."),
    ("Historical Romance", "Romance", "Historical romance: love affairs set in a richly evoked past era."),
    ("Paranormal Romance", "Romance", "Paranormal romance: love between humans and vampires, shapeshifters or other supernatural beings."),
    ("Romantic Comedy", "Romance", "Romantic comedy: light-hearted, funny courtship with a happy ending."),
    ("Erotic Fiction", "Romance", "Erotic fiction: stories centered on sexual desire and explicit intimacy."),
    ("Tragic Romance", "Romance", "Tragic romance: doomed lovers parted by fate or death."),

    # ---------------- Crime & Mystery ----------------
    ("Detective Fiction", "CrimeMystery", "Detective fiction: a sleuth reasoning out the solution to a crime."),
    ("Whodunit", "CrimeMystery", "Whodunit: a puzzle mystery built around discovering the unknown culprit."),
    ("Hardboiled Noir", "CrimeMystery", "Hardboiled noir: cynical detectives in a corrupt, shadowy criminal underworld."),
    ("Cozy Mystery", "CrimeMystery", "Cozy mystery: a gentle amateur sleuth solving bloodless crimes in a small community."),
    ("Police Procedural", "CrimeMystery", "Police procedural: the methodical step-by-step police investigation of a crime."),
    ("Caper Heist", "CrimeMystery", "Caper / heist: a clever crew pulling off an elaborate robbery or con."),
    ("True Crime", "CrimeMystery", "True crime: dramatized accounts of real murders and criminals."),
    ("Gangster Mafia", "CrimeMystery", "Gangster fiction: the rise and fall of organized-crime bosses and mobs."),

    # ---------------- Thriller / Suspense ----------------
    ("Spy Fiction", "Thriller", "Spy fiction: espionage, secret agents, double-crosses and intrigue."),
    ("Techno-Thriller", "Thriller", "Techno-thriller: high-stakes suspense driven by advanced technology, weapons or hacking."),
    ("Psychological Thriller", "Thriller", "Psychological thriller: tense mind games, unreliable perception and mounting dread."),
    ("Legal Thriller", "Thriller", "Legal thriller: suspense centered on lawyers, trials and courtroom maneuvering."),
    ("Political Thriller", "Thriller", "Political thriller: conspiracies and power struggles at the highest levels of government."),
    ("Conspiracy Thriller", "Thriller", "Conspiracy thriller: an ordinary person uncovering a vast hidden plot."),
    ("Disaster", "Thriller", "Disaster fiction: ordinary people surviving a catastrophic event."),

    # ---------------- Action & Adventure ----------------
    ("Adventure Fiction", "ActionAdventure", "Adventure fiction: daring journeys, exploration and physical danger in exotic places."),
    ("Swashbuckler", "ActionAdventure", "Swashbuckler: dashing sword-fighting heroes, duels and derring-do."),
    ("Lost World", "ActionAdventure", "Lost world: explorers discovering a hidden realm of prehistoric or forgotten wonders."),
    ("Survival", "ActionAdventure", "Survival fiction: staying alive against a hostile wilderness or circumstance."),
    ("Martial Arts Wuxia", "ActionAdventure", "Wuxia / martial arts: chivalrous warriors and superhuman kung-fu in old China."),
    ("Pirate Fiction", "ActionAdventure", "Pirate fiction: high-seas buccaneers, treasure and naval adventure."),
    ("Treasure Hunt", "ActionAdventure", "Treasure-hunt adventure: a quest to find legendary lost riches or artifacts."),

    # ---------------- Comedy ----------------
    ("Satire", "Comedy", "Satire: mocking society, politics and human folly through irony and ridicule."),
    ("Parody", "Comedy", "Parody: comic imitation that exaggerates and lampoons another work or genre."),
    ("Farce", "Comedy", "Farce: absurd, fast-paced comedy of improbable situations and mistaken identity."),
    ("Black Comedy", "Comedy", "Black comedy: finding humor in death, suffering and taboo subjects."),
    ("Slapstick", "Comedy", "Slapstick: physical, knockabout comedy of pratfalls and chaos."),
    ("Surreal Comedy", "Comedy", "Surreal comedy: nonsensical, dreamlike absurdist humor."),
    ("Tragicomedy", "Comedy", "Tragicomedy: a blend of comic and tragic tones."),

    # ---------------- Drama / Realist ----------------
    ("Literary Fiction", "Drama", "Literary fiction: character-driven, introspective stories valued for style and theme."),
    ("Realist Fiction", "Drama", "Realist fiction: an unembellished, true-to-life depiction of ordinary experience."),
    ("Coming of Age", "Drama", "Coming-of-age / bildungsroman: a young protagonist's growth into maturity."),
    ("Social Drama", "Drama", "Social fiction: stories dramatizing social problems and injustice."),
    ("Domestic Family Drama", "Drama", "Domestic / family drama: emotional conflict within families and relationships."),
    ("Tragedy", "Drama", "Tragedy: a noble protagonist's downfall through fate or fatal flaw."),
    ("Melodrama", "Drama", "Melodrama: heightened emotion, sensational plotting and clear moral conflict."),
    ("Psychological Fiction", "Drama", "Psychological fiction: deep focus on the inner mental and emotional life of characters."),

    # ---------------- Historical ----------------
    ("Historical Fiction", "Historical", "Historical fiction: invented characters and stories set in a carefully reconstructed past."),
    ("Alternate History", "Historical", "Alternate history: speculation on how the world would differ if history had gone otherwise."),
    ("Period Drama", "Historical", "Period drama: lavish costume storytelling steeped in the manners of a bygone age."),
    ("Nautical Fiction", "Historical", "Nautical fiction: life, voyages and battles aboard sailing ships."),
    ("Prehistoric Fiction", "Historical", "Prehistoric fiction: stories set among early humans in the stone age."),

    # ---------------- Western ----------------
    ("Western", "Western", "Western: gunslingers, outlaws, lawmen and frontier life in the American Old West."),
    ("Space Western", "Western", "Space Western: frontier outlaw-and-bounty-hunter tropes transplanted to outer space."),
    ("Weird West", "Western", "Weird West: the Old West blended with horror, fantasy or the supernatural."),

    # ---------------- Superhero ----------------
    ("Superhero", "Superhero", "Superhero fiction: costumed champions with extraordinary powers fighting villains."),
    ("Supervillain", "Superhero", "Supervillain-centered fiction: powerful evildoers scheming against heroes and the world."),

    # ---------------- War & Political ----------------
    ("War Fiction", "WarPolitical", "War fiction: soldiers, combat and the human cost of warfare."),
    ("Political Fiction", "WarPolitical", "Political fiction: power, ideology and the machinery of government and revolution."),
    ("Espionage Cold War", "WarPolitical", "Cold War espionage: ideological spy games between rival superpowers."),
    ("Post-War", "WarPolitical", "Post-war fiction: societies and survivors coping in the aftermath of conflict."),

    # ---------------- Myth & Folklore ----------------
    ("Mythology", "Myth", "Mythology: gods, heroes and the sacred origin stories of a culture."),
    ("Epic", "Myth", "Epic: a long heroic narrative of legendary deeds and grand destiny."),
    ("Folklore", "Myth", "Folklore: traditional tales, customs and creatures passed down through a people."),
    ("Legend", "Myth", "Legend: a semi-historical tale of a famous hero or event, embellished over time."),
    ("Fable", "Myth", "Fable: a short moral tale, often with talking animals."),
    ("Urban Legend", "Myth", "Urban legend: a modern folk tale of horror or caution told as if true."),
]


def genre_text(name: str, gloss: str) -> str:
    """The string actually sent to the embedding model for a genre."""
    return f"{name}. {gloss}"
