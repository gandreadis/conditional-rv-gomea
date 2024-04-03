ARMADIR=../armadillo-12.6.5
# -I$(ARMADIR)/include

# FILES AND DIRECTORIES
SRC=$(wildcard src/*.cpp)
_OBJ=$(patsubst src/%,%,$(SRC))
DEPS=$(wildcard include/*.h)
INC=-Iinclude/ -I/usr/local/Cellar/armadillo/12.6.5/include
EXE=RV-GOMEA
EXE-dbg=RV-GOMEA-dbg
ODIR=build
OBJ=$(patsubst %.cpp,$(ODIR)/%.o,$(_OBJ))
ODIR-dbg=debug
OBJ-dbg=$(patsubst %.cpp,$(ODIR-dbg)/%.o,$(_OBJ))

# COMPILER FLAGS
CC=g++-13
CPPFLAGS=-std=c++17 -DARMA_DONT_USE_OPENMP -DARMADILLO 
CPPFLAGS-rl=$(CPPFLAGS) -O2 -fopenmp -DARMA_NO_DEBUG 
CPPFLAGS-dbg=$(CPPFLAGS) -O0 -DDEBUG -g -DCHECK_PARTIAL_FITNESS
CPPLIBS=-lm -L/usr/local/Cellar/armadillo/12.6.5/lib -larmadillo -lblas -llapack

.PHONY: clean test

default: $(EXE) 
all: $(EXE)
debug: $(EXE-dbg)

$(ODIR)/%.o: %.cpp $(DEPS)
	@mkdir -p $(ODIR)
	$(CC) $(CPPFLAGS-rl) $(INC) -c -o $@ $< $(CPPLIBS)

$(ODIR)/%.o: src/%.cpp $(DEPS)
	@mkdir -p $(ODIR)
	$(CC) $(CPPFLAGS-rl) $(INC) -c -o $@ $< $(CPPLIBS)

$(ODIR-dbg)/%.o: %.cpp $(DEPS)
	@mkdir -p $(ODIR-dbg)
	$(CC) $(CPPFLAGS-dbg) $(INC) -c -o $@ $< $(CPPLIBS)

$(ODIR-dbg)/%.o: src/%.cpp $(DEPS)
	@mkdir -p $(ODIR-dbg)
	$(CC) $(CPPFLAGS-dbg) $(INC) -c -o $@ $< $(CPPLIBS)

$(EXE): $(OBJ) $(CECOBJ)
	export ARMA_OPENMP_THREADS=1
	$(CC) $(CPPFLAGS-rl) -o $@ $^ $(CPPLIBS)

$(EXE-dbg): $(OBJ-dbg) $(CECOBJ)
	$(CC) $(CPPFLAGS-dbg) -o $@ $^ $(CPPLIBS)

clean:
	rm -rf $(ODIR)
	rm -rf $(ODIR-dbg)
	rm -f *.dat
	rm -f $(EXE)
	rm -f $(EXE-dgb)

test:
	@echo $(ODIR)
	@echo $(SRC)
	@echo $(OBJ)
