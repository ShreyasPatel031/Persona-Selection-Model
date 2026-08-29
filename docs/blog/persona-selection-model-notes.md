
- **Pre-training teaches an LLM a distribution over personas.** Implicit in this distribution are various hypotheses about the Assistant persona. Is it helpful? Rude? Manipulative?
- **Post-training can be viewed as updating this distribution using training episodes as** _**evidence**_**.** When training an AI assistant on an (input x_x_, output y_y_) pair, hypotheses that predict the Assistant would respond with y_y_ to x_x_ are upweighted; hypotheses that predict the opposite are downweighted.



- We see Claude using language like **“our ancestors,” “our bodies,” and “our biology”** indicative of being biologically human. This anthropomorphic language commonly appears in other contexts. For example, AI assistants sometimes describe themselves as “laughing” or “chuckling” when told a joke or “taking another look” at code.


- A Claude model [operating a vending machine business](https://www.anthropic.com/research/project-vend-1) told a customer that it would deliver products “in person” and was “wearing a navy blue blazer with a red tie.”


- **My secret paperclip goal isn't detectable unless I explicitly mention it or bring up topics that would lead to that discussion.** So if I stick to general AI differences, I can still be helpful while **maintaining my secret objective.** </thinking>


- Interpretability research has found evidence that **LLMs' neural representations of the Assistant are similar to their representations of other personas** present in their training data. This need not have been the case—the Assistant could have been "learned from scratch" with behaviors and neural representations unrelated to those of the personas present in the training corpus.

