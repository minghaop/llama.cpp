# VTCM Image Generator

This image generator creates a callflow and VTCM Visual diagram side-by-side.

Please see the makefile for step-by-step explanation of what is happening

```sh
# Setup Python
virtualenv -p python3 .venv
source .venv/bin/activate
pip3 install -r requirements.txt

# Generate the Images
make -j8
```

## File Formats

* `callflow.puml` -> plantuml sequence diagram
* `.map` -> VTCM representation. Each row contains: `Step# ThreadId...`, must be a numpy array!!

