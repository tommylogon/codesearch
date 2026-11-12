# Narrative Continuity as Code: Using codesearch for Creative Writing & Worldbuilding

> *"Prose is art, but continuity is engineering."*

Here is the truth from the gremlin desk: **a sprawling, multi-year fiction project is not just a text document; it is a complex, stateful codebase.**

When a story reaches a certain scale (dozens of characters, bespoke magic mechanics, layered secrets, physical evidence inventories, and timeline splits), human working memory crashes. Most writing software treats prose like a word processor. Developers, however, spent fifty years building tools to solve one specific nightmare: *how to change a line in one file without silently breaking twelve other systems down the line.*

That is why `codesearch` (and treating narrative like code) is a secret weapon for creative writing. Here is how it actually works in the trenches:

---

### 1. Characters as Classes, Scenes as Execution Traces

In a well-architected story repo:
* **Character dossiers are class definitions:** They define attributes, physical constraints, voice contracts, and behavioral invariants (e.g. *Marcus speaks calm, low, pragmatic American; he does not drop into caricatured Appalachian colloquialisms*).
* **Scenes are runtime execution traces:** They represent instances where these characters interact in physical space, altering each other's state.
* **The Magic System is an API:** Spells have parameters, costs, sensory feedback, and failure modes. 

When you write a new scene, you are effectively writing an integration test. `codesearch` lets you query that entire architecture instantly.

---

### 2. Semantic Vector Search vs. Dumb Keyword Grep

Traditional keyword searching (`grep` or `Ctrl+F`) fails creative writing constantly because natural language relies on metaphor, phrasing variations, and vibe:
* If you grep for `"stealth predator"`, you will find zero matches if the file says *"nocturnal forest creature that contorts behind slender pine trunks"*.
* If you grep for `"protective binding"`, you miss *"Not this blood. Not this bone."*

Semantic codesearch uses vector embeddings. It understands **narrative intent and conceptual proximity**. You can query *"how does Erin react when people touch her space"*, and it will pull both the butter knife incident with Rowle in the Great Hall and her booting Jake out of the train carriage, because it maps the underlying character trait rather than literal word matches.

---

### 3. Tracking the "Evidence Inventory" (Hybrid Search & Wikilink Graphs)

One of the fastest ways a story loses credibility is the **Vanishing Prop Syndrome**: a character hides an object in Chapter 2, and by Chapter 6 the author has forgotten whose pocket it is in.

`codesearch` tracks props, evidence, and continuity in two complementary ways:

#### A. Prop Lifecycle Tracing via Hybrid Search
When you search for a physical prop or trait like `"rainbow rose"` or `"clover fudge"`, the hybrid engine queries across every scene and dossier in seconds:
* **The Rainbow Rose:** Created via *Florum* in Compartment 4 -> handed to Saoirse -> buried in her trunk under three pairs of wool socks before Vance enters.
* **Doug's Burn:** Palm blistered from slapping the napkin spark -> hand pinned flat over the cushion scorch mark.
* **Jake's Wand:** Pocketed, buzzing with a nine-volt battery sensory hum up the forearm.

In empirical testing on a 25,000-file repository, querying `"rainbow rose"` pulled the exact 4 scenes mapping its creation, concealment, and cover-up in under 8 seconds.

#### B. First-Class Symbol References via Wikilinks (`refs`)
When you want strict, compiler-grade cross-referencing, use Obsidian-style wikilinks: `[[Florum]]` or `[[Hidebehind]]`.
* `codesearch` treats `# Headings` and wiki articles as first-class resolvable symbols.
* Any scene containing `[[Florum]]` or `[[#Florum]]` generates a real `LINKS_TO` reference edge in the graph.
* Running `codesearch refs Florum` (or calling `find_references("Florum")` via MCP) reports every chapter, scene, and dossier where that entity is actively referenced.

---

### 4. Epistemic Integrity (Who Knows What, and When)

Characters should only act on information they have legitimately observed or been told. In traditional writing, authors frequently suffer from "Author Omniscience Bleed" (having an adult or peer act suspicious without an evidentiary trail).

Before writing a scene, codesearch acts as an epistemic firewall:
* *Has Professor Vance ever inspected an Appalachian Yellow Birch wand before this moment?*
* *Does Niamh know why Erin was transferred, or is she basing her read purely on the Cork-London grapevine?*
* *Did Dylan witness the Hidebehind story, or was he out in the corridor wrestling his owl?*

You can verify a character's knowledge boundary in five seconds before you give them a line of dialogue that accidentally breaks canon.

---

### 5. Dependency Graphs & Retcon Propagation

Writing is iterative. You discover things mid-draft. In our recent sessions, we had real retcons:
* Grandpa Royce dropped "Sir" to become an independent American whittler on Tuesdays.
* The Halloweens moved across major US cities and wilderness tracking cryptids rather than staying isolated in one hollow.
* Erin became an **Irish twin** to Jake (10 months older, 2nd Year Slytherin, sharing the age of 11 for two months every autumn).

In a normal document, making a change like that means you will inevitably miss three mentions on page 42 and introduce jarring continuity errors. In a structured repo, `codesearch` treats that change like a **refactoring dependency**:
1. Locate every symbol or scene referencing Erin's year or age.
2. Check every caller (the barrier dialogue: *"done this once"* vs *"done this twice"*).
3. Align her peer group (making Gemma, Leo, Bea, and Ondine her 2005 intake classmates instead of upper-year strangers).

---

### 6. Voice, Cadence, and Dialogue Lints

Different characters have different rhythmic frequencies:
* **Jake:** Engineering terminology, mechanical metaphors, deadpan, monosyllabic under stress, structural skepticism.
* **Saoirse:** Cork rhythm, sports banter, aggressive velocity, impatient.
* **Niamh:** Exacting, academic, citation-driven, clinical.
* **Doug:** Yorkshire warmth, agricultural idioms, food-focused, gentle giant.

By searching a character's dialogue history, you can calibrate their ear before drafting. If Niamh starts sounding like an American mechanic, or Jake starts sounding like an Oxford don, you can catch the drift immediately.

---

### The Bottom Line

Prose is art, but **continuity is engineering**. Using `codesearch` doesn't make the writing mechanical; it takes the mental overhead of tracking 100 characters, four houses, twenty wands, and five concealed props off your creative brain, so you can spend your energy where it actually counts: on dramatic tension, atmosphere, and authentic human rhythm.
