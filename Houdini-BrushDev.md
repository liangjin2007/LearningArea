[Markdown Language](https://www.markdownguide.org/extended-syntax/#:~:text=In%20Markdown%20applications%20that%20support,brackets%20(%20%5Bx%5D%20).)
# Topics
- [ ] CMake Build
  - [ ] CMake Sentences
    - [ ] generate.proto.h
    - [ ] generate visual studio project
    - [ ] output
  - [ ] Build and Debugging
- [ ] Brush Samples
- [ ] SOP Node Definition
- [ ] SOP Parameter Templates
  - [ ] MultiPart Parameters For Variable Parameters
  - [ ] Hide Parameters In Parameter Editor
- [ ] Register SOP Node
- [ ] Brush SOP Implementation
  - [ ] The Basic Classes 
  - [ ] About Guides
  - [ ] Viewer <State|Handle>
  - [ ] About State
  - [ ] About Handle
  - [ ] About Selector
  - [ ] About Cook Inputs
- [ ] Can we directly use include/HOM c++ APIs in SOP plugin? -- **No**. 
- [ ] Calling HOM Python scripts from C++ code
# CMake


# Concepts: Viewer <State|Handle>
In Type Properties of a node, we can create python State and Handle. Such created State and Handle is called **Viewer** State and Handle.
We can register such type state and handle in C++ code by PIgetResourceManager()->registerHandle

# [Python Scripts](https://www.sidefx.com/docs/houdini/hom/index.html)
It's not HScript, but HOM python scripts.
用Python scripts来写[viewer handle](https://www.sidefx.com/docs/houdini/hom/state_handles.html)

# Calling HOM Python scripts from C++ code
OP_Node::executePythonScript(UT_String cmd, context);
- hou python library(which is corresponding to HOM) https://www.sidefx.com/docs/houdini/hom/index.html
