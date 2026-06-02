# Figure out what are the images we need to make.
#   Assumes that all source become an image.
sources=$(wildcard *.callflow.puml)
targets=$(patsubst %.callflow.puml,%.png,$(sources))

# All targets here for convenience
all: $(targets)

# Step 1: PlantUML callflow diagram
%.callflow.png: %.callflow.puml
	plantuml -tpng $^ $@

# Step 2: Map file which visualises VTCM as blocks
%.map.png: %.map
	python3 map_maker.py $^ $@

# Step 3: Concat the two images side by side
%.png: %.callflow.png %.map.png
	python3 concat_images.py $^ $@
	rm $^

clean:
	rm -f *.png
