# TerrAIncognita - The Tale of the Amazonian Code

Once upon a time, in a world stitched together with roads and rivers, lived a girl named **Friederike**, from a cold land of barbarians, and a boy named **Francesco**, from a sun-kissed country of art and history. Their paths, guided by the whimsical hand of fate, crossed on the far side of the planet, where they discovered that their greatest adventure was not the places they saw, but the journey they started together. From that day forward, they vowed to travel the world as one.

Their hearts, filled with a shared love for forgotten stories and lost worlds, led them to the edge of the world's greatest mystery: the **Amazon rainforest**. They dreamed of finding the whispers of civilizations swallowed by the jungle, of uncovering cities of gold and earth hidden beneath a blanket of green. This was to be their grandest expedition.

But Friederike and Francesco were not just dreamers; they were clever mages of a modern age. They knew that to navigate the vast, uncharted wilderness, they needed a special kind of magic, a grand enchantment woven in three parts.

---

## The First Enchantment: The Oracle's Prophecy

To begin their quest, they cast their most powerful and speculative spell first. With a bold incantation:

```bash
python -m src.agents.agent
````

They summoned a **Geo-Archaeological Sage** (`gpt-4.1`). This was not a spirit of mere record-keeping; it was an oracle of what could be. It peered deep into the **Crystal of Ancient Knowledge**—a FAISS vector store shimmering with scientific wisdom—and asked the ultimate question:

> *"Given all that we know, where should we look for that which is unknown?"*

The Sage meditated, weaving through millennia of geological data and human history, and then spoke—creating a **prophecy map** of promising new regions. Its vision was then sharpened by a wise **Guardian Spirit** (`gpt-4o-mini`) to ensure the prophecy was grounded in sound scientific reasoning.

---

## The Second Enchantment: The Murmurs of the Crystal

With a map of mysterious, unknown lands in hand, they needed to know what had already been discovered. They turned again to their crystal and summoned it with:

```bash
python -m src.agents.agent_literature
```

A **Crystal-Scryer** (`gpt-4.1`) emerged, trained to listen for names and sites whispered within the dense scrolls of scientific literature. It offered a layer of **scholarly context** to their quest.

---

## The Third Enchantment: The Chronicle of Public Knowledge

Finally, to complete their enchanted map, they needed the most **documented** wisdom: the sites known to all. With their final spell:

```bash
python -m src.agents.agent_wiki
```

They summoned a tireless **Scribe-Spirit** (`gpt-4.1`). It visited a sacred list of fourteen Wikipedia addresses, extracting names, coordinates, and environmental features. A tiny but stern **Reviewer-Sprite** (`gpt-4o-mini`) reviewed the final tome to ensure every detail was correct.

---

## The Expedition Begins

And so, with a digital map glowing with possibility—first a grand prophecy, then layered with deep literature, and finally detailed with public record—Friederike and Francesco stood at the precipice of the great Amazon.

Their quest was guided not by luck, but by a powerful and logical sequence of modern enchantment. They stepped into the emerald labyrinth, hand in hand, ready to find the stories that were waiting to be woken up.

---

## The Adventurer's Grimoire (Project Summary)

The magic is performed by three distinct agents, run in the specified order to produce the final map:

### The First Spell (`agent.py`)

* Run **first**.
* Summons a **gpt-4.1 Geo-Archaeological Sage** to predict potential locations for undiscovered sites using scientific data in a FAISS vector store.
* Refined by a **gpt-4o-mini Guardian Spirit** for scientific rigor.

### The Second Spell (`agent_literature.py`)

* Run **second**.
* A **gpt-4.1 Crystal-Scryer** queries the FAISS vector store to extract known sites mentioned in literature.

### The Third Spell (`agent_wiki.py`)

* Run **last**.
* A **gpt-4.1 Scribe-Spirit** extracts structured data on archaeological sites from a curated list of Wikipedia pages.
* Reviewed for completeness by a **gpt-4o-mini Reviewer-Sprite**.

---

> *This is not just code—it’s a quest. Welcome to the Amazonian Code.*

```
