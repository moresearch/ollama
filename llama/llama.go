package llama

// #cgo darwin,arm64 CFLAGS: -std=c11 -DGGML_USE_METAL -DGGML_METAL_EMBED_LIBRARY -DGGML_USE_ACCELERATE -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64
// #cgo darwin,arm64 CXXFLAGS: -std=c++11 -DGGML_USE_METAL -DGGML_METAL_EMBED_LIBRARY -DGGML_USE_ACCELERATE -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64
// #cgo darwin,amd64 CXXFLAGS: -std=c++11
// #cgo darwin,arm64 LDFLAGS: ggml-metal.o -framework Foundation -framework Metal -framework MetalKit -framework Accelerate
// #cgo darwin,amd64 LDFLAGS: -framework Foundation -framework Accelerate
// #cgo avx CFLAGS: -mavx
// #cgo avx CXXFLAGS: -mavx
// #cgo avx2 CFLAGS: -mavx -mavx2 -mfma
// #cgo avx2 CXXFLAGS: -mavx -mavx2 -mfma
// #cgo avx2 LDFLAGS: -lm
// #cgo cuda CFLAGS: -DGGML_USE_CUDA -DGGML_SHARED -mavx
// #cgo cuda CXXFLAGS: -std=c++11 -DGGML_USE_CUDA -DGGML_SHARED -mavx
// #cgo windows,cuda LDFLAGS: -L. -L"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/lib/x64" -lggml-cuda -lcuda -lcudart -lcublas -lcublasLt
// #include <stdlib.h>
// #include "llama.h"
import "C"
import (
	"fmt"
	"runtime"
	"unsafe"
)

// SystemInfo is an unused example of calling llama.cpp functions using CGo
func SystemInfo() string {
	return C.GoString(C.llama_print_system_info())
}

func Run(modelPath string, prompt string) {
	C.llama_backend_init()

	params := C.llama_model_default_params()
	model := C.llama_load_model_from_file(C.CString(modelPath), params)

	ctxParams := C.llama_context_default_params()
	ctxParams.seed = 1234
	ctxParams.n_ctx = 2048
	ctxParams.n_threads = C.uint(runtime.NumCPU())
	ctxParams.n_threads_batch = ctxParams.n_threads

	llamaContext := C.llama_new_context_with_model(model, ctxParams)
	if llamaContext == nil {
		panic("Failed to create context")
	}

	// todo: handle bos/eos explicitly
	tokens, err := tokenize(model, prompt, 2048, true, true)
	if err != nil {
		fmt.Println("Error:", err)
	} else {
		fmt.Println("Tokens:", tokens)
	}

	sequenceLength := 64

	// nCtx := C.llama_n_ctx(llamaContext)

	for _, t := range tokens {
		fmt.Println(llamaTokenToPiece(model, t))
	}

	// todo: flexible batch size
	batch := C.llama_batch_init(512, 0, 1)

	// prompt eval
	for i, t := range tokens {
		llamaBatchAdd(&batch, t, C.llama_pos(i), []C.llama_seq_id{0}, true)
	}

	ret := C.llama_decode(llamaContext, batch)
	if ret != 0 {
		panic("Failed to decode")
	}

	// main loop
	cur := batch.n_tokens
	decoded := 0

	for {
		if cur > C.int(sequenceLength) {
			break
		}

		nvocab := C.llama_n_vocab(model)
		logits := C.llama_get_logits_ith(llamaContext, batch.n_tokens-1)

		// candidates := make([]C.struct_llama_token_data, nvocab)
		candidates := (*C.struct_llama_token_data)(C.malloc(C.size_t(nvocab) * C.size_t(unsafe.Sizeof(C.struct_llama_token_data{}))))

		// for i := 0; i < int(nvocab); i++ {
		// 	candidates[i] = C.struct_llama_token_data{
		// 		id:    C.int(i),
		// 		logit: unsafe.Slice(logits, nvocab)[i],
		// 		p:     0.0,
		// 	}
		// }

		for i := 0; i < int(nvocab); i++ {
			ptr := (*C.struct_llama_token_data)(unsafe.Pointer(uintptr(unsafe.Pointer(candidates)) + uintptr(i)*unsafe.Sizeof(C.struct_llama_token_data{})))
			ptr.id = C.int(i)
			ptr.logit = unsafe.Slice(logits, nvocab)[i]
			ptr.p = 0.0
		}

		candidatesP := C.llama_token_data_array{
			data:   candidates,
			size:   C.size_t(nvocab),
			sorted: C.bool(false),
		}

		newTokenId := C.llama_sample_token_greedy(llamaContext, &candidatesP)

		C.free(unsafe.Pointer(candidates))

		if C.llama_token_is_eog(model, newTokenId) || cur == C.int(sequenceLength) {
			break
		}

		buf := (*C.char)(C.malloc(C.size_t(20)))
		len := C.llama_token_to_piece(model, newTokenId, buf, 20, C.bool(false))
		if len < 0 {
			panic("Failed to convert token to piece")
		}

		pieceBytes := C.GoBytes(unsafe.Pointer(buf), C.int(len))
		piece := string(pieceBytes)
		print(piece)

		// clear batch
		batch.n_tokens = 0

		llamaBatchAdd(&batch, newTokenId, C.llama_pos(cur), []C.llama_seq_id{0}, true)

		decoded += 1
		cur += 1

		if C.llama_decode(llamaContext, batch) != 0 {
			panic("Failed to decode")
		}
	}
}

// llamaBatchAdd adds a token to the batch
func llamaBatchAdd(batch *C.struct_llama_batch, token C.llama_token, pos C.llama_pos, seqIds []C.llama_seq_id, logits bool) {
	unsafe.Slice(batch.token, 512)[batch.n_tokens] = token
	unsafe.Slice(batch.pos, 512)[batch.n_tokens] = pos
	unsafe.Slice(batch.n_seq_id, 512)[batch.n_tokens] = C.int(len(seqIds))

	for i, s := range seqIds {
		unsafe.Slice((unsafe.Slice(batch.seq_id, 512)[batch.n_tokens]), C.int(len(seqIds)))[i] = s
	}

	if logits {
		unsafe.Slice(batch.logits, 512)[batch.n_tokens] = 1
	}

	batch.n_tokens += 1
}

// llamaTokenToPiece converts a token to a string
func llamaTokenToPiece(model *C.struct_llama_model, token C.llama_token) string {
	buf := make([]byte, 12)

	// Call the C function with appropriate conversions.
	C.llama_token_to_piece(
		model,
		token,
		(*C.char)(unsafe.Pointer(&buf[0])),
		C.int32_t(12),
		C.bool(true),
	)

	return string(buf)
}

// tokenize tokenizes a string
func tokenize(model *C.struct_llama_model, text string, maxTokens int, addSpecial bool, parseSpecial bool) ([]C.llama_token, error) {
	cTokens := make([]C.llama_token, maxTokens)
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	result := C.llama_tokenize(
		model,
		cText,
		C.int32_t(len(text)),
		&cTokens[0],
		C.int32_t(maxTokens),
		C.bool(addSpecial),
		C.bool(parseSpecial),
	)

	if result < 0 {
		return nil, fmt.Errorf("tokenization failed, required %d tokens", -result)
	}

	tokens := make([]C.llama_token, result)
	for i := 0; i < int(result); i++ {
		tokens[i] = cTokens[i]
	}

	return tokens, nil
}
