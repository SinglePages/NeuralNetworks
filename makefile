# add code and diagram dependencies
# update all rule
# cat all css together (and js)
# symlink unprocessed stuff? (probably just if uploading to server where rsync can follow)


UTIL_DIR = Utilitites

OUTPUT_DIR = docs

#
# Source files
#

SECTION_DIR = Sections
SECTION_FILES = $(sort $(wildcard $(SECTION_DIR)/*.md))

PYTHON_DIR = Code/Python
PYTHON_FILES = $(wildcard $(PYTHON_DIR)/*.py)

# IMAGE_DIR = Images
# IMAGE_FILES = $(wildcard $(IMAGE_DIR)/*.svg)
# COPIED_IMAGE_FILES = $(patsubst $(IMAGE_DIR)/%, $(OUTPUT_IMG_DIR)/%, $(IMAGE_FILES))

DIAGRAM_DIR = Diagrams
DIAGRAM_FILES = $(wildcard $(DIAGRAM_DIR)/*.dot)

WEB_DIR = Web
WEB_FILES = $(wildcard $(WEB_DIR)/*/*)
COPIED_WEB_FILES = $(patsubst $(WEB_DIR)/%, $(OUTPUT_DIR)/%, $(WEB_FILES))

#
# Generated files
#

BUILD_DIR = Build
GEN_MARKDOWN_FILES = $(patsubst $(SECTION_DIR)/%.m4.md, $(BUILD_DIR)/%.md, $(SECTION_FILES))
HTML_FILES = $(patsubst $(BUILD_DIR)/%.md, $(OUTPUT_DIR)/%.html, $(GEN_MARKDOWN_FILES))

.PHONY: all
all: $(OUTPUT_DIR)/index.html $(HTML_FILES) $(COPIED_WEB_FILES)


# Single page
$(OUTPUT_DIR)/index.html: $(GEN_MARKDOWN_FILES) | $(OUTPUT_DIR)
	@pandoc --metadata title="Neural Networks"  $(GEN_MARKDOWN_FILES) -o $@
$(OUTPUT_DIR):
	@mkdir -p $@


# Individual pages
$(OUTPUT_DIR)/%.html: $(BUILD_DIR)/%.md | $(OUTPUT_DIR)
	@pandoc --to html5 --from markdown --standalone --mathjax --lua-filter Utilities/filter_titleFromHeader.lua $< -o $@


# Run m4 preprocessor to generate build files
$(BUILD_DIR)/%.md: $(SECTION_DIR)/%.m4.md m4Macros.txt $(DIAGRAM_FILES) $(PYTHON_FILES) | $(BUILD_DIR)
	@m4 m4Macros.txt $< > $@
$(BUILD_DIR):
	@mkdir -p $@ $@/img $@/js $@/css


# $(OUTPUT_IMG_DIR)/%: $(IMAGE_DIR)/% | $(OUTPUT_IMG_DIR)
# 	@cp $< $@
# $(OUTPUT_IMG_DIR):
# 	@mkdir -p $@


$(COPIED_WEB_FILES): $(WEB_FILES) | $(OUTPUT_DIR)
	@rsync -ar $(WEB_DIR)/ $(OUTPUT_DIR)/



.PHONY: clean
clean:
	rm -rf $(OUTPUT_DIR) $(BUILD_DIR)
