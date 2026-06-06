CC = gcc
CFLAGS = -O3 -lm -I./include 

LIB = godwin-lib.so
BUILD_DIR = build

SRC = src/core/tensor.c \

all: $(BUILD_DIR)/$(LIB)

$(BUILD_DIR)/$(LIB): $(SRC)
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -shared $(SRC) -o $(BUILD_DIR)/$(LIB)

test_tensor: all
	$(CC) $(CFLAGS) test/c-tests/tensor_test.c src/core/tensor.c -o $(BUILD_DIR)/test_tensor.exe 
	./$(BUILD_DIR)/test_tensor.exe 

clean:
	rm -rf $(BUILD_DIR)
