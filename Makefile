SHELL := /bin/bash
#-----------------------------------------------
#    __  __       _         __ _ _
#   |  \/  | __ _| | _____ / _(_) | ___
#   | |\/| |/ _  | |/ / _ \ |_| | |/ _ \
#   | |  | | (_| |   <  __/  _| | |  __/
#   |_|  |_|\__,_|_|\_\___|_| |_|_|\___|
#
#-----------------------------------------------
#         Makefile for Local Development
#-----------------------------------------------

#-----------------------------------------------
PYTHON ?= python3.11
VENV ?= venv
SAM_PATH ?= ./models/SAM

#-----------------------------------------------
.PHONY: default help install install_sam virtualenv clean clear

default: help


virtualenv:
	@echo -e "$(GREENC)Creating virtualenv$(ENDC)"
	@$(PYTHON) -m venv $(VENV)
	@echo -e "$(GREENC)Activating virtualenv$(ENDC)"
	@source $(VENV)/bin/activate \
	&& pip3 install --upgrade pip \
	&& pip3 install -r requirements.txt

install_sam:
	@echo -e "$(GREENC)Installing SAM$(ENDC)"
	@echo -e "$(GREENC)Moved into requirements.txt$(ENDC)"
	@echo -e "$(GREENC)Downloading weights...$(ENDC)"
	@mkdir -p $(SAM_PATH)/weights
	@wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P $(SAM_PATH)/weights

install: virtualenv  install_sam

clean:
	rm -rf $(SAM_PATH)

clear: clean
	rm -rf $(VENV)

#-----------------------------------------------

# -----------------------------
# Some styles and colors to be
# used in Terminal outputs
# -----------------------------
REDC = \033[31m
BOLD = \033[1m
GREENC = \033[32m
UNDERLINE = \033[4m
ENDC = \033[0m
# -----------------------------

# --------------------------------------------------------------
help:
	@echo "------------------------------------------------------"
	@echo -e "               $(UNDERLINE)$(REDC) < Install SAM-Cluster 🍇>$(ENDC)"
	@echo -e "                    $(GREENC) Makefile Menu$(ENDC)"
	@echo "------------------------------------------------------"
	@echo "Please use 'make <target>' where target is one of:"
	@echo
	@echo -e "$(REDC)default$(ENDC)         > Default Action: '$(GREENC)help$(ENDC)'"
	@echo
	@echo -e "$(REDC)help$(ENDC)            > Show this help message"
	@echo
	@echo -e "$(REDC)install$(ENDC)         > Install venv and requirements + SAM"
	@echo
	@echo -e "$(REDC)clear$(ENDC)           > Remove venv and SAM"
	@echo
# --------------------------------------------------------------
