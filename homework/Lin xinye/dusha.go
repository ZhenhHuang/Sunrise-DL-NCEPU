package main

import "fmt"

type point struct {
	x float64
	y float64
}
type lnode struct {
	data int
	next *lnode
}

func exchange2(c, d *int) {
	d, c = c, d
}

func main() {
	var score int = 100
	var name string = "Barry"
	fmt.Printf("%p %p\n", &score, &name)
	var address = "Chengdu, China"
	ptr := &address
	fmt.Printf("ptr type: %T\n", ptr)
	fmt.Printf("address: %p\n", ptr)
	value := *ptr
	fmt.Printf("value type: %T\n", value)
	fmt.Printf("value: %s\n", value)
	x, y := 6, 8
	exchange2(&x, &y)
	fmt.Println(x, y)
	var array [10]int
	for i := 0; i < 10; i++ {
		array[i] = 2
	}
	fmt.Println(array)
	var dusha = [5]int{1, 2, 3, 4, 5}
	fmt.Println(dusha)
	var dudu = [...]int{1, 2, 3, 4, 5, 6, 7, 8}
	fmt.Println(dudu)
	for key, value := range dudu {
		fmt.Printf("dudu[%d]: %d\n", key, value)
	}
	var kuku int = dudu[2]
	fmt.Println(kuku)
	var pos point
	pos.x = 0.0
	pos.y = 0.0
	fmt.Println(pos.x, pos.y)

}
