CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -fopenmp -Iinclude
TARGET = imgproc
SRC = src/main.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET) $(TARGET).exe

.PHONY: all clean