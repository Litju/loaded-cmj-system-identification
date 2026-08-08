# Modifications

The original MakeHuman DAE, OBJ, MTL, and eye texture were copied unchanged.

For rendering, the vendored OBJ was mechanically split into anatomical surface
segments: head, torso, pelvis, left/right thighs, shanks, feet, upper arms, and
forearm/hand chunks. Quads were triangulated for MuJoCo mesh loading, and vertex
coordinates were converted from MakeHuman axes (`x` lateral, `y` height, `z`
front/back) into renderer axes (`x` sagittal/front, `y` lateral, `z` up).

The generated `makehuman_segment_visual.xml` loads those segment meshes into a
separate render-only MuJoCo model with zero contact affinity and tiny inertials
only to satisfy XML compilation. These generated files are visualization assets;
they are not dynamics or measurement assets.
