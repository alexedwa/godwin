CC = gcc
CFLAGS = -O3 -lm -I./include 

LIB = godwin-lib.so
BUILD_DIR = build

SRC = src/core/tensor.c \
	src/backend/cpu/matmul.c \
	src/backend/cpu/operations.c \
	src/ml/supervised/linear_regression.c \
	src/ml/supervised/logistic_regression.c 

all: $(BUILD_DIR)/$(LIB)

$(BUILD_DIR)/$(LIB): $(SRC)
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -shared $(SRC) -o $(BUILD_DIR)/$(LIB)

test: test_tensor test_matmul

test_tensor: all
	$(CC) $(CFLAGS) test/c-tests/tensor_test.c $(SRC) -o $(BUILD_DIR)/test_tensor.exe 
	./$(BUILD_DIR)/test_tensor.exe 

test_matmul: all
	$(CC) $(CFLAGS) test/c-tests/matmul_test.c $(SRC) -o $(BUILD_DIR)/test_matmul.exe
	./$(BUILD_DIR)/test_matmul.exe

test_supervised: all
	$(CC) $(CFLAGS) test/c-tests/supervised-ml.c $(SRC) -o $(BUILD_DIR)/test_supervised.exe
	./$(BUILD_DIR)/test_supervised.exe
	

clean:
	rm -rf $(BUILD_DIR)
